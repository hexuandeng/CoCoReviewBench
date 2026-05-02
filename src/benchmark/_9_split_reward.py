import os
import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from itertools import combinations
from collections import Counter  # 新增

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# ---------------- trackers ----------------
@dataclass
class ScoreTracker:
    counts: Dict[float, int] = field(default_factory=lambda: {-1.0: 0, 0.0: 0, 1.0: 0})
    total: int = 0
    def update(self, score: float):
        if score not in self.counts:
            self.counts[score] = 0
        self.counts[score] += 1
        self.total += 1
        if self.total and self.total % 100 == 0:
            logger.info(self.pretty())
    def get_ratio(self) -> Dict[float, float]:
        if self.total == 0:
            return {k: 0.0 for k in self.counts}
        return {k: v / self.total for k, v in self.counts.items()}
    def pretty(self) -> str:
        r = self.get_ratio()
        ks = sorted(self.counts.keys())
        parts = [f"total={self.total}"] + [f"{k:+g}: {self.counts[k]}/{r[k]:.2%}" for k in ks]
        return " | ".join(parts)

global_tracker = ScoreTracker()
batch_tracker  = ScoreTracker()

# --------------- helpers -----------------
_LINE_RE = re.compile(r"^\s*(?:Part|ID)\s*[:#]?\s*(\d+)\s*[:.\-–]\s*(.+?)\s*$", re.IGNORECASE)

def _strip_think(s: str) -> str:
    return re.sub(r"(?is)<think>.*?</think>", "", s or "").strip()

def _parse_labels_field(raw: str) -> List[str]:
    """
    允许多标签：'Label 2, 5' / '2,5' / 'Label N/A'
    返回规范化后的标签字符串列表（数字或'N/A'），重复自动去重。
    """
    s = raw.strip()
    if s.lower().startswith("label"):
        s = s[5:].strip()
    labs = [x.strip() for x in re.split(r"[,\s]+", s) if x.strip()]
    out, seen = [], set()
    for x in labs:
        xx = x.upper()
        if xx in {"N/A", "NA"}:
            tag = "N/A"
        elif re.fullmatch(r"\d+", x):
            tag = str(int(x))  # 去前导零
        else:
            logger.debug("ignore non-numeric/non-NA token: %r", x)
            continue
        if tag not in seen:
            out.append(tag); seen.add(tag)
    return out

def _parse_answer_to_membership(text: str, num_parts: int) -> Tuple[Optional[List[set]], List[str]]:
    """
    解析模型回答为 membership: List[Set[int]]，长度=num_parts，
    每个位置是该 part 的“数字标签集合”（N/A 不计入集合）。
    返回 (membership or None, errors)
    - 缺行/越界/重复行 -> 直接 None（视为格式错误）
    """
    raw = _strip_think(text)
    logger.info(raw)
    if not raw:
        return None, ["empty prediction"]

    out: Dict[int, set] = {}
    errors: List[str] = []

    for line in raw.splitlines():
        if not line.strip():
            continue
        m = _LINE_RE.match(line)
        if not m:
            errors.append(f"bad line format: {line!r}")
            continue
        k = int(m.group(1))
        if not (1 <= k <= num_parts):
            errors.append(f"part id out of range: {k}")
            continue
        if k in out:
            errors.append(f"duplicate line for Part {k}")
            continue
        labs = _parse_labels_field(m.group(2))
        s = {int(x) for x in labs if x != "N/A"}  # N/A -> 空集合
        out[k] = s

    # 覆盖必须恰好 1..num_parts
    if len(out) != num_parts:
        missed = [i for i in range(1, num_parts + 1) if i not in out]
        if missed:
            errors.append(f"missing parts: {missed[:20]}{' ...' if len(missed) > 20 else ''}")
        return None, errors

    mem = [out[i] for i in range(1, num_parts + 1)]
    return mem, errors

# ---------------- Omega Index（重叠聚类） ----------------
def _pair_comembership_counts(partition_sets, universe):
    """
    统计给定重叠分区在 universe 元素集上的“对的共同隶属个数”：
      返回：
        pair2c: dict[(i,j) -> c]，c 是 (i,j) 在多少个簇里共同出现（i<j）
        hist:   Counter，键=j 表示有多少对的共同隶属个数为 j（不含 j=0）
    """
    pair2c = {}
    hist = Counter()
    for S in partition_sets:
        T = sorted(S & universe)
        if len(T) < 2:
            continue
        for i, j in combinations(T, 2):
            key = (i, j)
            pair2c[key] = pair2c.get(key, 0) + 1
    for c in pair2c.values():
        hist[c] += 1
    return pair2c, hist

def omega_index_overlapping(part1_sets, part2_sets, universe):
    """
    Collins & Dent (1988) 的 Omega Index：
      Omega = (u - e) / (1 - e)
      其中：
        - u：实测一致率（两方案对 (i,j) 的共同隶属个数 c1 与 c2 是否相等）
        - e：独立假设的期望一致率，e = sum_j (P1_j * P2_j) / P^2
             其中 Pk_j 为方案 k 中“共同隶属个数= j”的对数目，P=C(n,2)
    """
    n = len(universe)
    if n < 2:
        return 1.0  # 少于2个元素，视为完美一致

    P = n * (n - 1) // 2  # 总对数

    c1, h1_nonzero = _pair_comembership_counts(part1_sets, universe)
    c2, h2_nonzero = _pair_comembership_counts(part2_sets, universe)

    # 各自“共同隶属=0”的对数
    h1 = dict(h1_nonzero)
    h2 = dict(h2_nonzero)
    h1[0] = P - sum(h1_nonzero.values())
    h2[0] = P - sum(h2_nonzero.values())

    # 实测一致率 u：c1==c2 的对的比例
    keys1 = set(c1.keys())
    keys2 = set(c2.keys())
    union_pairs = keys1 | keys2
    both_zero = P - len(union_pairs)  # 两边都为0的对
    equal_nonzero = sum(1 for k in (keys1 & keys2) if c1[k] == c2[k])
    M = both_zero + equal_nonzero
    u = M / P

    # 期望一致率 e：按分布独立相乘
    all_js = set(h1.keys()) | set(h2.keys())
    e = sum(h1.get(j, 0) * h2.get(j, 0) for j in all_js) / (P * P)

    denom = (1.0 - e)
    if denom == 0.0:
        return 1.0 if u == 1.0 else 0.0
    return (u - e) / denom

# --------------- VERL entry ----------------
def compute_score(
    solution_str: str,
    ground_truth: str,
    *args, **kwargs
) -> float:
    """
    新评分（与你的方案一致）：
      final_score = omega_non_na + na_acc + format_score

      - omega_non_na：仅在“双方共同认为非 N/A”的元素上，根据 Omega Index 计算（范围约 [-1,1]）
      - na_acc：逐句 N/A 准确率（范围 [0,1]）
      - format_score：格式正确性（完全无错误=1.0，否则=0.0）
    失败策略：
      - 若预测解析失败（缺行/越界/重复导致覆盖不全），返回 omega=-1.0, na_acc=0.0, format_score=0.0 → final=-1.0
    """
    # 取 extra_info
    extra: Optional[Dict[str, Any]] = None
    if "example" in kwargs and isinstance(kwargs["example"], dict):
        extra = kwargs["example"].get("extra_info") or kwargs["example"].get("extra") or None
    if extra is None:
        extra = kwargs.get("extra_info", None)
    if not isinstance(extra, dict):
        global_tracker.update(-1.0); batch_tracker.update(-1.0)
        return -1.0

    num_parts = extra.get("num_parts", None)
    gold_parsed = extra.get("pred_parsed", None)
    pred_mem, pred_errs = _parse_answer_to_membership(solution_str, num_parts)

    # ---------------- 格式正确性 ----------------
    format_ok = (pred_mem is not None) and (not pred_errs)
    format_score = 0.0 if format_ok else -1.0

    # 若解析失败，直接返回 0（严格按照你的失败策略）
    if format_score == -1.0:
        omega_non_na = 0.0
        final_score = -0.5
        logger.info("[SCORE] n=%d | omega=%.4f | fmt=%.1f | final=%.4f | errs=%s",
                    num_parts, omega_non_na, format_score, final_score, "; ".join(pred_errs))
        global_tracker.update(final_score)
        batch_tracker.update(final_score)
        return final_score

    gold_sets = [set(s) for s in gold_parsed]
    label2set: Dict[int, set] = {}
    for pid, labset in enumerate(pred_mem, start=1):
        for lab in labset:
            label2set.setdefault(lab, set()).add(pid)
    pred_sets = list(label2set.values())

    gold_clean = set().union(*gold_sets) if gold_sets else set()
    pred_clean = set().union(*pred_sets) if pred_sets else set()
    common_clean = gold_clean & pred_clean

    logger.info(str(gold_sets))
    logger.info(str(pred_sets))
    omega_non_na = omega_index_overlapping(gold_sets, pred_sets, common_clean)

    # ---------- 聚合 ----------
    final_score = omega_non_na + format_score
    final_score = max(-0.5, min(1.0, float(final_score)))

    # 日志
    def _s(x): return _strip_think(x)[:100].replace("\n", "\\n")
    logger.info("[SCORE]\t%s\t n=%d\t omega=%.4f\t fmt=%.1f\t final=%.4f",
                _s(solution_str), num_parts, omega_non_na, format_score, final_score)
    if pred_errs:
        logger.info("[FORMAT_ERRS] %s", "; ".join(pred_errs))

    global_tracker.update(final_score)
    batch_tracker.update(final_score)
    return final_score
