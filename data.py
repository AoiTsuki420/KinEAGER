import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import set_seed,KineticsScaler


class EnzymeDataset(Dataset):
    def __init__(self, 
                 df: pd.DataFrame, 
                 scaler=None, 
                 supervised=True,
                 prot_struct=None,
                 lig_struct=None,
                 return_row_id: bool = False, #默认 False 保持老格式
                 source_ids=None,
                 source_col: str | None = None,
                 source_map: dict | None = None,
                 default_source_id: int = 0,
                 phys_cols: list[str] | None = None,
                 phys_mask_col: str | None = "phys_mask",
                 phys_quality_col: str | None = "phys_quality",
                ):
        """
        df: DataFrame 包含 kcat(s^-1)、Km(M)、Sequence、Smiles
        scaler: KineticsScaler 实例，必须先 fit
        """
        self.df_raw = df
        self.df = df.reset_index(drop=True)
        self.row_ids = df.index.to_numpy()          # 原始 row_id（与 NPZ 对齐）
        self.return_row_id = return_row_id

        self.supervised = supervised
        self.scaler = scaler
        self.phys_cols = phys_cols
        self.phys_mask_col = phys_mask_col
        self.phys_quality_col = phys_quality_col

        self.source_ids = None
        n = len(self.df)
        if source_ids is not None:
            src_arr = np.asarray(source_ids)
            if src_arr.shape[0] != n:
                raise ValueError(f"source_ids 长度不匹配: got {src_arr.shape[0]}, expected {n}")
            self.source_ids = src_arr.astype(np.int64)
        elif source_col is not None and source_col in self.df_raw.columns:
            src_series = self.df_raw[source_col]
            if np.issubdtype(src_series.dtype, np.number):
                self.source_ids = src_series.to_numpy().astype(np.int64)
            else:
                mapper = source_map or {
                    "catapro": 0,
                    "cataprodata": 0,
                    "catpred-kcat": 0,
                    "catpred_kcat": 0,
                    "catpredkcat": 0,
                    "skid-kcat": 1,
                    "skid_kcat": 1,
                    "skidkcat": 1,
                    "catpred-km": 1,
                    "catpred_km": 1,
                    "catpredkm": 1,
                    "skid-km": 2,
                    "skid_km": 2,
                    "skidkm": 2,
                }
                src_vals = []
                for v in src_series.astype(str).to_list():
                    key = v.strip().lower().replace(" ", "").replace("_", "-")
                    src_vals.append(int(mapper.get(key, default_source_id)))
                self.source_ids = np.asarray(src_vals, dtype=np.int64)
        else:
            self.source_ids = np.full((n,), int(default_source_id), dtype=np.int64)

        self.seq_col = None
        for c in ["Sequence", "sequence", "Protein Sequence"]:
            if c in df.columns:
                self.seq_col = c
                break
        if self.seq_col is None:
            raise ValueError("df 缺少序列列（支持: Sequence/sequence/Protein Sequence）")

        self.smi_col = None
        for c in ["Smiles", "SMILES", "Substrate SMILES", "Substrate_SMILES"]:
            if c in df.columns:
                self.smi_col = c
                break
        if self.smi_col is None:
            raise ValueError("df 缺少底物 SMILES 列（支持: Smiles/SMILES/Substrate SMILES）")

        kcat_col = None
        for c in ["kcat(s^-1)", "kcat", "kcat_value"]:
            if c in df.columns:
                kcat_col = c
                break

        km_col = None
        for c in ["Km(M)", "Km", "Km_value"]:
            if c in df.columns:
                km_col = c
                break

        if supervised and (kcat_col is not None or km_col is not None):
            if scaler is None:
                raise ValueError("For supervised training, a fitted KineticsScaler must be provided.")
            
            if kcat_col is None:
                kcat = np.full((len(df),), np.nan, dtype=np.float32)
            else:
                kcat = pd.to_numeric(df[kcat_col], errors="coerce").to_numpy(dtype=np.float32)

            if km_col is None:
                Km = np.full((len(df),), np.nan, dtype=np.float32)
            else:
                Km = pd.to_numeric(df[km_col], errors="coerce").to_numpy(dtype=np.float32)

            self.labels = self.scaler.transform(kcat, Km)
            
        else:
            self.labels = None
            if supervised:
                raise ValueError(
                    "Supervised=True 但 df 中缺少 kcat/Km 列。"
                )

        if self.phys_cols is None:
            self.phys_cols = sorted(
                [c for c in self.df.columns if c.startswith("phys_") and c not in {"phys_mask", "phys_quality"}]
            )
        self.use_phys = len(self.phys_cols) > 0
        self.phys_feat = None
        self.phys_mask = None
        self.phys_quality = None
        if self.use_phys:
            self.phys_feat = self.df[self.phys_cols].astype(np.float32).to_numpy()
            if self.phys_mask_col and self.phys_mask_col in self.df.columns:
                self.phys_mask = pd.to_numeric(self.df[self.phys_mask_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            else:
                self.phys_mask = np.isfinite(self.phys_feat).all(axis=1).astype(np.float32)
            if self.phys_quality_col and self.phys_quality_col in self.df.columns:
                self.phys_quality = pd.to_numeric(self.df[self.phys_quality_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            else:
                self.phys_quality = self.phys_mask.copy()
            self.phys_feat = np.nan_to_num(self.phys_feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            self.phys_mask = np.clip(self.phys_mask, 0.0, 1.0).astype(np.float32)
            self.phys_quality = np.clip(self.phys_quality, 0.0, 1.0).astype(np.float32)
        
        self.prot_struct = None
        self.lig_struct = None
        self.use_struct = False
        
        self.struct_index_by_row_id = False
        n = len(self.df)
        
        max_row_id = int(self.row_ids.max()) if len(self.row_ids) > 0 else -1
        self.struct_index_by_row_id = False

        if (prot_struct is None) ^ (lig_struct is None):
            raise ValueError("prot_struct 和 lig_struct 要么都提供，要么都为 None，当前只有一个为 None。")

        if prot_struct is not None:
            if isinstance(prot_struct, np.ndarray):
                prot_t = torch.from_numpy(prot_struct.astype(np.float32))
            elif isinstance(prot_struct, torch.Tensor):
                prot_t = prot_struct.float()
            else:
                raise TypeError("prot_struct 必须是 np.ndarray 或 torch.Tensor，或 None")

            if isinstance(lig_struct, np.ndarray):
                lig_t = torch.from_numpy(lig_struct.astype(np.float32))
            elif isinstance(lig_struct, torch.Tensor):
                lig_t = lig_struct.float()
            else:
                raise TypeError("lig_struct 必须是 np.ndarray 或 torch.Tensor，或 None")
                
            prot_n = prot_t.shape[0]
            lig_n = lig_t.shape[0]
            
            if prot_n != lig_n and (prot_n == n) != (lig_n == n):
                raise ValueError(
                    f"prot_struct 与 lig_struct 行数不一致或对齐模式不一致："
                    f"prot_n={prot_n}, lig_n={lig_n}, len(df)={n}"
                )
            if prot_n == n and lig_n == n:
                self.struct_index_by_row_id = False
            else:
                if prot_n <= max_row_id:
                    raise ValueError(
                        f"prot_struct 行数 {prot_n} 不足以覆盖 df 最大 row_id={max_row_id}"
                    )
                if lig_n <= max_row_id:
                    raise ValueError(
                        f"lig_struct 行数 {lig_n} 不足以覆盖 df 最大 row_id={max_row_id}"
                    )
                self.struct_index_by_row_id = True

            self.prot_struct = prot_t.cpu()
            self.lig_struct = lig_t.cpu()
            self.use_struct = True


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row[self.seq_col]
        smi = row[self.smi_col]
        row_id = int(self.row_ids[idx]) #原始行号（用来从 NPZ 取结构向量）
        source_id = int(self.source_ids[idx]) if self.source_ids is not None else 0
        
        if self.supervised:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            label_tensor = None

        if not self.use_struct:
            if self.use_phys:
                phys_feat = torch.from_numpy(self.phys_feat[idx])
                phys_mask = torch.tensor(self.phys_mask[idx], dtype=torch.float32)
                phys_quality = torch.tensor(self.phys_quality[idx], dtype=torch.float32)
                if self.return_row_id:
                    return seq, smi, row_id, source_id, phys_feat, phys_mask, phys_quality, label_tensor
                return seq, smi, source_id, phys_feat, phys_mask, phys_quality, label_tensor
            if self.return_row_id:
                return seq, smi, row_id, source_id, label_tensor  # (5)
            else:
                return seq, smi, source_id, label_tensor  # (4)

        if self.struct_index_by_row_id:
            p_vec = self.prot_struct[row_id]
            l_vec = self.lig_struct[row_id]
        else:
            p_vec = self.prot_struct[idx]
            l_vec = self.lig_struct[idx]

        if self.return_row_id:
            if self.use_phys:
                phys_feat = torch.from_numpy(self.phys_feat[idx])
                phys_mask = torch.tensor(self.phys_mask[idx], dtype=torch.float32)
                phys_quality = torch.tensor(self.phys_quality[idx], dtype=torch.float32)
                return seq, smi, row_id, source_id, p_vec, l_vec, phys_feat, phys_mask, phys_quality, label_tensor
            return seq, smi, row_id, source_id, p_vec, l_vec, label_tensor  # (7)
        else:
            if self.use_phys:
                phys_feat = torch.from_numpy(self.phys_feat[idx])
                phys_mask = torch.tensor(self.phys_mask[idx], dtype=torch.float32)
                phys_quality = torch.tensor(self.phys_quality[idx], dtype=torch.float32)
                return seq, smi, source_id, p_vec, l_vec, phys_feat, phys_mask, phys_quality, label_tensor
            return seq, smi, source_id, p_vec, l_vec, label_tensor  # (6)
        
def make_dataloader(
    df: pd.DataFrame, 
    scaler=None, 
    batch_size=8, 
    shuffle=True, 
    supervised=True,
    prot_struct=None,
    lig_struct=None,
    return_row_id: bool = False,
    source_ids=None,
    source_col: str | None = None,
    source_map: dict | None = None,
    default_source_id: int = 0,
    phys_cols: list[str] | None = None,
    phys_mask_col: str | None = "phys_mask",
    phys_quality_col: str | None = "phys_quality",
    ):
    """
    df: DataFrame 或已读取的 CSV 数据
    scaler: KineticsScaler 实例，必须先 fit
    """
    if supervised and scaler is None:
        raise ValueError("For supervised dataloader, a fitted KineticsScaler must be provided.")

    dataset = EnzymeDataset(
        df,
        scaler=scaler,
        supervised=supervised,
        prot_struct=prot_struct,
        lig_struct=lig_struct,
        return_row_id=return_row_id,
        source_ids=source_ids,
        source_col=source_col,
        source_map=source_map,
        default_source_id=default_source_id,
        phys_cols=phys_cols,
        phys_mask_col=phys_mask_col,
        phys_quality_col=phys_quality_col,
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


EPS = 1e-12


class SingleTaskEnzymeDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        target: str = "kcat",
        prot_struct=None,
        lig_struct=None,
        return_row_id: bool = False,
    ):
        """
        单任务版 Dataset，支持结构向量。

        df: 至少包含列：
            - 'Sequence'
            - 'Smiles'
            - 对应任务的标签列：
                target='kcat' -> 'kcat(s^-1)'
                target='Km'   -> 'Km(M)'

            允许 df 里有 NaN，本类会自动丢弃该任务没有标签的行。

        prot_struct / lig_struct:
            - 可选结构向量数组，shape (N_all, D_p)/(N_all, D_l)
            - 这里 N_all 必须等于 df 原始行数（过滤前）
            - 本类会用同一个布尔 mask 过滤 df 和两个结构数组，保证对齐
            - 类型可以是 np.ndarray 或 torch.Tensor
        """
        if target not in ("kcat", "Km"):
            raise ValueError("target 必须是 'kcat' 或 'Km'")
        self.df_raw = df
        self.row_ids_all = df.index.to_numpy()
        self.return_row_id = return_row_id
        df_work = df.reset_index(drop=True)
        n_all = len(df_raw)

        if target == "kcat":
            if "kcat(s^-1)" not in df_raw.columns:
                raise ValueError("df 中缺少列 'kcat(s^-1)'")
            mask_valid = ~df_raw["kcat(s^-1)"].isna().values  # (N_all,)
        else:  # Km
            if "Km(M)" not in df_raw.columns:
                raise ValueError("df 中缺少列 'Km(M)'")
            mask_valid = ~df_raw["Km(M)"].isna().values

        if mask_valid.sum() == 0:
            raise ValueError(f"target={target} 时，df 中没有任何有效标签")

        self.df = df_work[mask_valid].reset_index(drop=True)
        self.row_ids = self.row_ids_all[mask_valid]
        self.target = target

        if target == "kcat":
            y_raw = df["kcat(s^-1)"].astype(float).values  # (N_task,)
            y_log = np.log10(y_raw + EPS).astype(np.float32)
        else:
            y_raw = df["Km(M)"].astype(float).values
            y_log = np.log10(y_raw * 1000.0 + EPS).astype(np.float32)

        self.labels = y_log  # (N_task,)

        self.prot_struct = None
        self.lig_struct = None
        self.use_struct = False
        self.struct_index_by_row_id = False
        
        n= len(self.df)
        max_row_id = int(self.row_ids.max()) if len(self.row_ids) > 0 else -1

        if (prot_struct is None) ^ (lig_struct is None):
            raise ValueError("prot_struct 和 lig_struct 要么都提供，要么都为 None")

        if prot_struct is not None:
            if isinstance(prot_struct, np.ndarray):
                prot_t = torch.from_numpy(prot_struct.astype(np.float32))
            elif isinstance(prot_struct, torch.Tensor):
                prot_t = prot_struct.float()
            else:
                raise TypeError("prot_struct 必须是 np.ndarray 或 torch.Tensor")

            if isinstance(lig_struct, np.ndarray):
                lig_t = torch.from_numpy(lig_struct.astype(np.float32))
            elif isinstance(lig_struct, torch.Tensor):
                lig_t = lig_struct.float()
            else:
                raise TypeError("lig_struct 必须是 np.ndarray 或 torch.Tensor")

            prot_n = prot_t.shape[0]
            lig_n = lig_t.shape[0]

            if prot_n != lig_n and (prot_n == n) != (lig_n == n):
                raise ValueError(
                    f"prot_struct 与 lig_struct 行数不一致或对齐模式不一致："
                    f"prot_n={prot_n}, lig_n={lig_n}, len(df_filtered)={n}"
                )

            if prot_n == n and lig_n == n:
                self.struct_index_by_row_id = False
            else:
                if prot_n <= max_row_id:
                    raise ValueError(f"prot_struct 行数 {prot_n} 不足以覆盖最大 row_id={max_row_id}")
                if lig_n <= max_row_id:
                    raise ValueError(f"lig_struct 行数 {lig_n} 不足以覆盖最大 row_id={max_row_id}")
                self.struct_index_by_row_id = True

            self.prot_struct = prot_t.cpu()
            self.lig_struct = lig_t.cpu()
            self.use_struct = True
            
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row["Sequence"]
        smi = row["Smiles"]
        row_id = int(self.row_ids[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.float32)  # 标量

        if not self.use_struct:
            if self.return_row_id:
                return seq, smi, row_id, y  # (4)
            else:
                return seq, smi, y  # (3)

        if self.struct_index_by_row_id:
            p_vec = self.prot_struct[row_id]
            l_vec = self.lig_struct[row_id]
        else:
            p_vec = self.prot_struct[idx]
            l_vec = self.lig_struct[idx]

        if self.return_row_id:
            return seq, smi, row_id, p_vec, l_vec, y  # (6)
        else:
            return seq, smi, p_vec, l_vec, y  # (5)

def make_single_task_dataloader(
    df: pd.DataFrame,
    target: str = "kcat",
    batch_size: int = 32,
    shuffle: bool = True,
    prot_struct=None,
    lig_struct=None,
    return_row_id: bool = False,
):
    dataset = SingleTaskEnzymeDataset(
        df,
        target=target,
        prot_struct=prot_struct,
        lig_struct=lig_struct,
        return_row_id=return_row_id,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
