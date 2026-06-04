
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any, Literal

import numpy as np
import pandas as pd
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]
AblationMode = Literal["none", "real", "random"]


@dataclass(frozen=True)
class FinalContactConfig:
    npz_path: Union[str, Path]
    prot_key: str = "prot_struct"
    lig_key: str = "lig_struct"
    expected_prot_dim: Optional[int] = 45
    expected_lig_dim: Optional[int] = 135
    dtype: Any = np.float32


@dataclass(frozen=True)
class FinalContactAblationConfig:
    """
    消融配置：
      - mode="none"   : 不提供结构向量（prot_struct=None, lig_struct=None），作为 baseline
      - mode="real"   : 使用 NPZ 中的真实结构向量
      - mode="random" : 使用与真实结构同形状的“固定随机向量表”
                        （强烈推荐：用于反驳“提升只是因为参数/容量变多”）
    """
    mode: AblationMode = "real"

    random_seed: int = 123
    random_distribution: Literal["normal", "uniform"] = "normal"
    random_scale: float = 1.0

    random_scope: Literal["global"] = "global"


class FinalContactStore:
    """
    从 NPZ 加载并提供结构向量服务：
      - prot_struct: (N, Dp)
      - lig_struct : (N, Dl)

    【对齐方式（推荐）】
      - 用 df.index 作为 row_id 对齐 NPZ 向量。
        关键要求：切分 train/val 时保留原始 index，不要 reset_index(drop=True)。

    【可选对齐方式】
      - 如果 NPZ 中有 ids 字段，可用 id -> row_id 映射对齐。

    【消融接口】
      - set_ablation(mode="none"/"real"/"random")
      - get_batch_by_rowids_torch(...)：给 batch 的 row_ids 返回 torch 结构向量（或 None）
    """

    def __init__(self, cfg: FinalContactConfig):
        self.cfg = cfg
        self._prot: Optional[np.ndarray] = None
        self._lig: Optional[np.ndarray] = None
        self._ids: Optional[np.ndarray] = None  # NPZ 可选字段 ids

        self._abl: FinalContactAblationConfig = FinalContactAblationConfig(mode="real")
        self._prot_rand: Optional[np.ndarray] = None
        self._lig_rand: Optional[np.ndarray] = None

    @property
    def prot(self) -> np.ndarray:
        if self._prot is None:
            raise RuntimeError("FinalContactStore 尚未 load()。请先调用 .load()")
        return self._prot

    @property
    def lig(self) -> np.ndarray:
        if self._lig is None:
            raise RuntimeError("FinalContactStore 尚未 load()。请先调用 .load()")
        return self._lig

    @property
    def shapes(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return self.prot.shape, self.lig.shape

    @property
    def ablation(self) -> FinalContactAblationConfig:
        return self._abl

    def set_ablation(self, abl: FinalContactAblationConfig) -> "FinalContactStore":
        """
        设置消融模式。
        推荐至少跑三组：
          - none   : baseline（不加结构）
          - random : 容量对照（同维度随机结构）
          - real   : 真实结构（目标）
        """
        self._abl = abl
        self._prot_rand = None
        self._lig_rand = None
        return self

    def load(self, mmap_mode: Optional[str] = None) -> "FinalContactStore":
        """
        加载 NPZ 并做严格校验。
        """
        npz_path = Path(self.cfg.npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"找不到 NPZ：{npz_path}")

        data = np.load(npz_path, allow_pickle=False, mmap_mode=mmap_mode)
        if self.cfg.prot_key not in data or self.cfg.lig_key not in data:
            raise KeyError(
                f"NPZ 必须包含键 '{self.cfg.prot_key}' 和 '{self.cfg.lig_key}'；"
                f"当前 keys={list(data.keys())}"
            )

        prot = np.asarray(data[self.cfg.prot_key], dtype=self.cfg.dtype)
        lig = np.asarray(data[self.cfg.lig_key], dtype=self.cfg.dtype)

        if prot.ndim != 2 or lig.ndim != 2:
            raise ValueError(f"prot/lig 必须是二维数组。prot={prot.shape}, lig={lig.shape}")
        if prot.shape[0] != lig.shape[0]:
            raise ValueError(f"N 不一致：prot N={prot.shape[0]}，lig N={lig.shape[0]}")

        if self.cfg.expected_prot_dim is not None and prot.shape[1] != self.cfg.expected_prot_dim:
            raise ValueError(f"prot 维度不匹配：got {prot.shape[1]}，expected {self.cfg.expected_prot_dim}")
        if self.cfg.expected_lig_dim is not None and lig.shape[1] != self.cfg.expected_lig_dim:
            raise ValueError(f"lig 维度不匹配：got {lig.shape[1]}，expected {self.cfg.expected_lig_dim}")

        ids = None
        if "ids" in data:
            ids = np.asarray(data["ids"])
            if ids.shape[0] != prot.shape[0]:
                raise ValueError(f"ids 长度不匹配：ids N={ids.shape[0]}，prot N={prot.shape[0]}")

        self._prot = prot
        self._lig = lig
        self._ids = ids

        self._prot_rand = None
        self._lig_rand = None
        return self

    def validate_for_df_by_rowid(self, df: pd.DataFrame) -> None:
        """
        校验 df.index 是否可以作为 row_id 去索引 NPZ 向量。

        必须满足：
          - df.index 是整数类型
          - 0 <= min(index)
          - max(index) < N_all
        """
        if self._prot is None or self._lig is None:
            raise RuntimeError("请先 load() 再 validate。")

        if len(df) == 0:
            return

        idx = df.index.to_numpy()
        if idx.dtype.kind not in ("i", "u"):
            raise TypeError(
                "df.index 必须是整数类型（用于 row_id 对齐）。"
                "请不要 reset_index(drop=True) 之后再丢失原始行号。"
            )

        n_all = self._prot.shape[0]
        mn, mx = int(idx.min()), int(idx.max())
        if mn < 0 or mx >= n_all:
            raise ValueError(
                f"df.index 超出 NPZ 向量范围：index 范围 [{mn},{mx}]，但 N_all={n_all}。"
                "请确保 df 的 index 保留自原始全量 df（与 NPZ 一致）。"
            )

    def get_by_rowids(self, row_ids: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """
        输入 row_ids（B,），输出 numpy：(prot[B, Dp], lig[B, Dl])
        """
        if self._prot is None or self._lig is None:
            raise RuntimeError("请先 load() 再 get_by_rowids。")

        if isinstance(row_ids, torch.Tensor):
            row_ids = row_ids.detach().cpu().numpy()
        row_ids = np.asarray(row_ids)
        if row_ids.ndim != 1:
            row_ids = row_ids.reshape(-1)
        if row_ids.dtype.kind not in ("i", "u"):
            raise TypeError(f"row_ids 必须是整数数组，当前 dtype={row_ids.dtype}")

        return self._prot[row_ids], self._lig[row_ids]

    def build_id_to_rowid(self) -> Dict[Any, int]:
        """
        若 NPZ 中存在 ids，则构建 id -> row_id 映射。
        """
        if self._ids is None:
            raise RuntimeError("NPZ 不包含 ids 字段，无法做 id 对齐。")
        return {
            (self._ids[i].item() if hasattr(self._ids[i], "item") else self._ids[i]): i
            for i in range(len(self._ids))
        }

    def get_by_ids(self, ids: ArrayLike, id_to_rowid: Optional[Dict[Any, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        输入 ids（B,），输出 numpy：(prot[B, Dp], lig[B, Dl])
        """
        if id_to_rowid is None:
            id_to_rowid = self.build_id_to_rowid()

        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().numpy()
        ids = np.asarray(ids)
        if ids.ndim != 1:
            ids = ids.reshape(-1)

        row_ids = np.array(
            [id_to_rowid[x.item() if hasattr(x, "item") else x] for x in ids],
            dtype=np.int64,
        )
        return self.get_by_rowids(row_ids)

    def _ensure_random_cache(self) -> None:
        """
        为 random 模式生成并缓存一张固定随机表（N_all x D）。
        """
        if self._prot is None or self._lig is None:
            raise RuntimeError("random 模式需要先 load() 真实 NPZ（用其 N 和 D）。")

        if self._prot_rand is not None and self._lig_rand is not None:
            return

        n_all, dp = self._prot.shape
        _, dl = self._lig.shape

        rng = np.random.default_rng(self._abl.random_seed)

        if self._abl.random_distribution == "normal":
            prot = rng.standard_normal(size=(n_all, dp)).astype(np.float32)
            lig = rng.standard_normal(size=(n_all, dl)).astype(np.float32)
            if self._abl.random_scale != 1.0:
                prot *= float(self._abl.random_scale)
                lig *= float(self._abl.random_scale)
        elif self._abl.random_distribution == "uniform":
            s = float(self._abl.random_scale)
            prot = rng.uniform(low=-s, high=s, size=(n_all, dp)).astype(np.float32)
            lig = rng.uniform(low=-s, high=s, size=(n_all, dl)).astype(np.float32)
        else:
            raise ValueError(f"未知 random_distribution={self._abl.random_distribution}")

        self._prot_rand = prot
        self._lig_rand = lig

    def _select_arrays_for_mode(self, mode: AblationMode) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        根据消融模式选择要返回的结构向量表：
          - none   -> (None, None)
          - real   -> (prot, lig)
          - random -> (prot_rand, lig_rand)
        """
        if mode == "none":
            return None, None
        if mode == "real":
            return self.prot, self.lig
        if mode == "random":
            self._ensure_random_cache()
            assert self._prot_rand is not None and self._lig_rand is not None
            return self._prot_rand, self._lig_rand
        raise ValueError(f"未知 mode={mode}")

    def to_torch(
        self,
        prot_np: np.ndarray,
        lig_np: np.ndarray,
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        numpy -> torch.float32
        """
        prot_t = torch.from_numpy(np.asarray(prot_np, dtype=np.float32))
        lig_t = torch.from_numpy(np.asarray(lig_np, dtype=np.float32))
        if device is not None:
            prot_t = prot_t.to(device=device, non_blocking=non_blocking)
            lig_t = lig_t.to(device=device, non_blocking=non_blocking)
        return prot_t, lig_t

    def get_batch_by_rowids_torch(
        self,
        row_ids: ArrayLike,
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = True,
        mode: Optional[AblationMode] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        【训练循环核心接口（建议用这个）】

        输入：
          row_ids: batch 的原始行号（df.index），形状 (B,)
          mode: None 表示使用 self.ablation.mode；否则覆盖

        输出：
          - mode="none"   -> (None, None)
          - mode="real"   -> (prot_struct[B, Dp], lig_struct[B, Dl]) torch tensor
          - mode="random" -> (prot_rand[B, Dp], lig_rand[B, Dl]) torch tensor

        用法：
          p_s, l_s = store.get_batch_by_rowids_torch(row_ids, device=device)
          out = model(seqs, smiles, prot_struct=p_s, lig_struct=l_s, meta=meta, use_mask=True)
        """
        mode = self._abl.mode if mode is None else mode
        prot_arr, lig_arr = self._select_arrays_for_mode(mode)
        if prot_arr is None or lig_arr is None:
            return None, None

        if isinstance(row_ids, torch.Tensor):
            row_ids_np = row_ids.detach().cpu().numpy()
        else:
            row_ids_np = np.asarray(row_ids)

        row_ids_np = row_ids_np.reshape(-1).astype(np.int64)
        
        n = prot_arr.shape[0]
        row_ids_np = row_ids_np.astype(np.int64)

        ok = (row_ids_np >= 0) & (row_ids_np < n)

        if not ok.all():
            if mode == "real":
                prot_np = np.zeros((len(row_ids_np), prot_arr.shape[1]), dtype=prot_arr.dtype)
                lig_np  = np.zeros((len(row_ids_np), lig_arr.shape[1]),  dtype=lig_arr.dtype)
                prot_np[ok] = prot_arr[row_ids_np[ok]]
                lig_np[ok]  = lig_arr[row_ids_np[ok]]
            elif mode == "random":
                prot_np = np.random.standard_normal((len(row_ids_np), prot_arr.shape[1])).astype(prot_arr.dtype)
                lig_np  = np.random.standard_normal((len(row_ids_np), lig_arr.shape[1])).astype(lig_arr.dtype)
            else:  # none
                return None, None
        else:
            prot_np = prot_arr[row_ids_np]
            lig_np  = lig_arr[row_ids_np]

        return self.to_torch(prot_np, lig_np, device=device, non_blocking=non_blocking)

    def get_batch_by_df_index_torch(
        self,
        df: pd.DataFrame,
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = True,
        mode: Optional[AblationMode] = None,
        validate: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        便捷接口：直接用 df.index 作为 row_ids（推荐）。
        """
        if validate:
            self.validate_for_df_by_rowid(df)
        row_ids = df.index.to_numpy()
        return self.get_batch_by_rowids_torch(
            row_ids=row_ids,
            device=device,
            non_blocking=non_blocking,
            mode=mode,
        )


def load_final_contact_npz(
    npz_path: Union[str, Path],
    expected_prot_dim: Optional[int] = 45,
    expected_lig_dim: Optional[int] = 135,
    prot_key: str = "prot_struct",
    lig_key: str = "lig_struct",
    mmap_mode: Optional[str] = None,
) -> FinalContactStore:
    """
    便捷加载函数。
    """
    cfg = FinalContactConfig(
        npz_path=npz_path,
        prot_key=prot_key,
        lig_key=lig_key,
        expected_prot_dim=expected_prot_dim,
        expected_lig_dim=expected_lig_dim,
    )
    return FinalContactStore(cfg).load(mmap_mode=mmap_mode)


"""
store = load_final_contact_npz("/root/autodl-fs/data/skid_kcat_struct_vecs_adv.npz")

store.set_ablation(FinalContactAblationConfig(mode="none"))
store.set_ablation(FinalContactAblationConfig(mode="random", random_seed=42, random_distribution="normal", random_scale=1.0))
store.set_ablation(FinalContactAblationConfig(mode="real"))

store.validate_for_df_by_rowid(df_train)
store.validate_for_df_by_rowid(df_val)

"""
