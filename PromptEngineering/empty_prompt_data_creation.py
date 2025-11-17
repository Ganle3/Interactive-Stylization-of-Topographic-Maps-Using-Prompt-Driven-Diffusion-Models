"""
Export random map tiles (no prompt, empty prompt JSONL)
- Export target & source images with same filename
- target saved in 'target/', source saved in 'source/'
- filenames start with 'empty_' and sequential from START_ID
"""

import os, json, random, time, traceback
from qgis.core import (
    QgsProject, QgsMapSettings, QgsMapRendererParallelJob, QgsRectangle,
    QgsSpatialIndex, QgsFeatureRequest, QgsVectorLayer, QgsUnitTypes
)
from PyQt5.QtGui import QImage, QColor
from PyQt5.QtCore import QSize

# ===================== 配置区 ======================
OUT_DIR = r"D:\Junyhuang\Project2_Data\Training Data\Item_color"
SRC_SUBDIR   = "source"
TGT_SUBDIR   = "target"
META_SUBDIR  = "meta"
JSONL_NAME   = "pairs.jsonl"

N_TILES      = 500
START_ID     = 9740            # 起始编号
WIDTH, HEIGHT = 512, 512
DPI          = 96
SCALE        = 5000
RANDOM_SEED  = 42
MAX_ATTEMPTS = 20000

GT_LAYERS = [
    "Dissolved_Eisenbahn",
    "Dissolved_Strasse",
    "DKM25_EINZELBAUM",
    "DKM25_GEBAEUDE",
    "DKM25_GEWAESSER_LIN",
    "DKM25_GEWAESSER_PLY",
    "DKM25_BODENBEDECKUNG",
]

SEG_LAYERS = [
    "DKM25_EISENBAHN",
    "DKM25_STRASSE",
    "DKM25_EINZELBAUM_Seg",
    "DKM25_GEBAEUDE_Seg",
    "DKM25_GEWAESSER_LIN_Seg",
    "DKM25_GEWAESSER_PLY_Seg",
    "DKM25_BODENBEDECKUNG_Seg",
]

random.seed(RANDOM_SEED)

# ===================== 工具函数 ======================
def layer_by_name(name):
    lst = QgsProject.instance().mapLayersByName(name)
    if not lst:
        raise RuntimeError(f"未找到图层：{name}")
    return lst[0]

def ensure_dirs(root):
    """确保 source / target / meta 目录存在"""
    src = os.path.join(root, SRC_SUBDIR); os.makedirs(src, exist_ok=True)
    tgt = os.path.join(root, TGT_SUBDIR); os.makedirs(tgt, exist_ok=True)
    meta = os.path.join(root, META_SUBDIR); os.makedirs(meta, exist_ok=True)
    return src, tgt, meta

def calc_map_units_size(px_w, px_h, dpi, scale):
    mu_w = (px_w / dpi) * (0.0254 * scale)
    mu_h = (px_h / dpi) * (0.0254 * scale)
    return mu_w, mu_h

def rect_from_center(cx, cy, w, h):
    return QgsRectangle(cx - w/2, cy - h/2, cx + w/2, cy + h/2)

def visible_layers_in_draw_order():
    proj = QgsProject.instance()
    order = proj.layerTreeRoot().layerOrder()
    vis = []
    for lyr in order:
        node = proj.layerTreeRoot().findLayer(lyr.id())
        if node and node.isVisible():
            vis.append(lyr)
    return vis

def pick_reference_layer(candidates):
    by_name = {lyr.name(): lyr for lyr in candidates}
    for n in GT_LAYERS:
        if n in by_name and isinstance(by_name[n], QgsVectorLayer):
            return by_name[n]
    for lyr in candidates:
        if isinstance(lyr, QgsVectorLayer):
            return lyr
    return None

def build_sindex(vlayer: QgsVectorLayer):
    feats = list(vlayer.getFeatures(QgsFeatureRequest().setNoAttributes()))
    if not feats:
        return None, []
    sidx = QgsSpatialIndex()
    for f in feats:
        sidx.addFeature(f)
    return sidx, feats

def any_feature_intersects(layers, rect):
    for lyr in layers:
        if isinstance(lyr, QgsVectorLayer):
            req = QgsFeatureRequest(rect)
            req.setNoAttributes()
            it = lyr.getFeatures(req)
            if next(it, None) is not None:
                return True
    return False

def export_map(extent: QgsRectangle, out_png: str, layers):
    """导出地图截图"""
    ms = QgsMapSettings()
    ms.setLayers(layers)
    ms.setOutputDpi(DPI)
    ms.setOutputSize(QSize(WIDTH, HEIGHT))
    ms.setExtent(extent)

    job = QgsMapRendererParallelJob(ms)
    job.start()
    job.waitForFinished()
    img = job.renderedImage()
    img.save(out_png, "PNG")

# ===================== 主逻辑 ======================
def main():
    proj = QgsProject.instance()
    crs = proj.crs()
    if crs.mapUnits() != QgsUnitTypes.DistanceMeters:
        print("[WARN] 工程 CRS 不是米单位，窗口尺寸可能不正确。")

    layers = visible_layers_in_draw_order()
    if not layers:
        raise RuntimeError("未发现可见图层，请先勾选可见。")

    ref = pick_reference_layer(layers)
    if ref is None:
        raise RuntimeError("未找到参考矢量层。")

    sidx, feats = build_sindex(ref)
    full_extent = ref.extent()
    w_mu, h_mu = calc_map_units_size(WIDTH, HEIGHT, DPI, SCALE)
    src_dir, tgt_dir, meta_dir = ensure_dirs(OUT_DIR)
    jsonl_path = os.path.join(meta_dir, JSONL_NAME)

    f = open(jsonl_path, "a", encoding="utf-8")

    found, tries = 0, 0
    while found < N_TILES and tries < MAX_ATTEMPTS:
        tries += 1
        if feats:
            feat = random.choice(feats)
            bb = feat.geometry().boundingBox()
        else:
            bb = full_extent
        if bb.isEmpty():
            continue

        cx = random.uniform(bb.xMinimum(), bb.xMaximum())
        cy = random.uniform(bb.yMinimum(), bb.yMaximum())
        rect = rect_from_center(cx, cy, w_mu, h_mu)

        if not full_extent.contains(rect): continue
        if not any_feature_intersects(layers, rect): continue

        sid = f"keep_{START_ID + found:06d}"
        out_tgt_png = os.path.join(tgt_dir, f"{sid}.png")
        out_src_png = os.path.join(src_dir, f"{sid}.png")

        try:
            export_map(rect, out_tgt_png, [layer_by_name(n) for n in GT_LAYERS])
            export_map(rect, out_src_png, [layer_by_name(n) for n in SEG_LAYERS])
        except Exception as e:
            print(f"[FAIL] 导出失败（{sid}）：{e}")
            traceback.print_exc()
            continue

        rec = {
            "id": sid,
            "source": os.path.relpath(out_src_png, OUT_DIR).replace("\\", "/"),
            "target": os.path.relpath(out_tgt_png, OUT_DIR).replace("\\", "/"),
            "prompt": "Keep original color."
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        found += 1
        if found % 25 == 0:
            print(f"[{found}/{N_TILES}] 进度 OK")

    f.close()
    print(f"\n✅ 完成：导出 {found} 张；JSON 追加：{jsonl_path}")
    
    
main()