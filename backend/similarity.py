import cv2
import numpy as np
import time


def color_texture_similarity(img_a, mask_a, img_b, mask_b):
    """
    颜色纹理相似度：衡量两个区域内颜色的"跳跃程度"差异

    原理：
    - 自然背景：相邻像素颜色渐变，梯度小且分布集中
    - 垃圾/遗留物：相邻像素颜色突变，梯度大且分布分散

    子项：
    1. 颜色跳跃强度（梯度均值）：整体颜色变化有多剧烈
    2. 颜色跳跃混乱度（梯度方差）：颜色变化是否杂乱无章
    3. 颜色丰富度（HSV色相分散度）：用了多少种不同的颜色

    返回: 综合分数(0~1), 子项详情dict
    """
    def preprocess(img_bgr):
        # 3x3均值滤波去除极值/噪点
        img_blurred = cv2.blur(img_bgr, (3, 3))
        # 转HSV + 光照归一化
        hsv = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
        return hsv.astype(np.float32)

    def compute_color_grad(hsv):
        h_ch = hsv[:, :, 0]
        s_ch = hsv[:, :, 1]
        h_grad_x = cv2.Sobel(h_ch, cv2.CV_64F, 1, 0, ksize=3)
        h_grad_y = cv2.Sobel(h_ch, cv2.CV_64F, 0, 1, ksize=3)
        s_grad_x = cv2.Sobel(s_ch, cv2.CV_64F, 1, 0, ksize=3)
        s_grad_y = cv2.Sobel(s_ch, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(h_grad_x**2 + h_grad_y**2 + s_grad_x**2 + s_grad_y**2), h_ch

    def region_color_features(color_grad, h_ch, mask):
        pixels = color_grad[mask > 0]
        h_pixels = h_ch[mask > 0]

        if pixels.size == 0:
            return 0.0, 0.0, 0.0

        # 子项1：颜色跳跃强度（梯度均值）
        jump_intensity = np.mean(pixels)

        # 子项2：颜色跳跃混乱度（梯度方差）
        jump_chaos = np.std(pixels)

        # 子项3：颜色丰富度（色相分散度）
        hist, _ = np.histogram(h_pixels, bins=18, range=(0, 180))
        hist = hist[hist > 0]
        prob = hist / hist.sum()
        color_entropy = -np.sum(prob * np.log2(prob))
        color_richness = color_entropy / np.log2(18)

        return jump_intensity, jump_chaos, color_richness

    hsv_a = preprocess(img_a)
    hsv_b = preprocess(img_b)

    grad_a, h_a = compute_color_grad(hsv_a)
    grad_b, h_b = compute_color_grad(hsv_b)

    feat_a = region_color_features(grad_a, h_a, mask_a)
    feat_b = region_color_features(grad_b, h_b, mask_b)

    # 计算三个子项的差异
    def safe_diff(va, vb):
        max_val = max(abs(va), abs(vb), 1e-6)
        return abs(va - vb) / max_val

    intensity_diff = safe_diff(feat_a[0], feat_b[0])
    chaos_diff = safe_diff(feat_a[1], feat_b[1])
    richness_diff = safe_diff(feat_a[2], feat_b[2])

    # 三个子项等权合并
    combined = (intensity_diff + chaos_diff + richness_diff) / 3.0

    details = {
        "jump_intensity_a": feat_a[0],
        "jump_intensity_b": feat_b[0],
        "intensity_diff": intensity_diff,
        "jump_chaos_a": feat_a[1],
        "jump_chaos_b": feat_b[1],
        "chaos_diff": chaos_diff,
        "color_richness_a": feat_a[2],
        "color_richness_b": feat_b[2],
        "richness_diff": richness_diff,
    }

    return min(1.0, combined), details


def edge_similarity_canny(gray_a, mask_a, gray_b, mask_b):
    """
    Canny 边缘相似度，包含：
    1. 边缘密度差异
    2. 边缘规则度差异（直线占比）
    返回 综合分数(0~1), (edges_a, edges_b), 子项详情dict
    """
    edges_a_full = cv2.Canny(gray_a, 70, 150)
    edges_b_full = cv2.Canny(gray_b, 70, 150)

    # 分别取各自 mask 区域的边缘
    edges_a = cv2.bitwise_and(edges_a_full, mask_a)
    edges_b = cv2.bitwise_and(edges_b_full, mask_b)

    # ---- 1. 边缘密度 ----
    area_a = cv2.countNonZero(mask_a)
    area_b = cv2.countNonZero(mask_b)

    if area_a == 0 or area_b == 0:
        return 1.0, (edges_a_full, edges_b_full), {}

    density_a = cv2.countNonZero(edges_a) / area_a
    density_b = cv2.countNonZero(edges_b) / area_b

    max_density = max(density_a, density_b, 1e-6)
    density_diff = abs(density_a - density_b) / max_density

    # ---- 2. 边缘规则度（直线像素占边缘像素的比例） ----
    def line_regularity(edge_mask):
        """
        用霍夫变换检测直线段，计算"直线上的像素数 / 总边缘像素数"
        返回 0~1，越高越规则
        """
        edge_count = cv2.countNonZero(edge_mask)
        if edge_count == 0:
            return 0.0

        # HoughLinesP 返回线段端点
        lines = cv2.HoughLinesP(edge_mask,
                                rho=1,
                                theta=np.pi / 180,
                                threshold=15,       # 最少多少票才算一条线
                                minLineLength=10,   # 最短线段长度
                                maxLineGap=5)       # 允许的断裂间距

        if lines is None:
            return 0.0

        # 把所有检测到的线段画到空白图上，数像素
        line_mask = np.zeros_like(edge_mask)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 1)

        # 只算和实际边缘重合的部分（防止虚假膨胀）
        overlap = cv2.bitwise_and(line_mask, edge_mask)
        line_pixels = cv2.countNonZero(overlap)

        return min(1.0, line_pixels / edge_count)

    reg_a = line_regularity(edges_a)
    reg_b = line_regularity(edges_b)

    max_reg = max(reg_a, reg_b, 1e-6)
    regularity_diff = abs(reg_a - reg_b) / max_reg

    # ---- 3. 综合 ----
    # 密度差和规则度差各占一半
    combined = density_diff * 0.5 + regularity_diff * 0.5

    details = {
        "density_a": density_a,
        "density_b": density_b,
        "density_diff": density_diff,
        "regularity_a": reg_a,
        "regularity_b": reg_b,
        "regularity_diff": regularity_diff,
    }

    return min(1.0, combined), (edges_a_full, edges_b_full), details


def area_comprehensive_similarity(img_a, mask_a, img_b, mask_b):
    """
    综合评判函数
    img_a, mask_a: 第一张图和对应mask
    img_b, mask_b: 第二张图和对应mask（大小可以不一致）
    """
    start_time = time.perf_counter()

    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    ct_score, ct_details = color_texture_similarity(img_a, mask_a, img_b, mask_b)
    e_score, edges_imgs, e_details = edge_similarity_canny(gray_a, mask_a, gray_b, mask_b)

    # 权重配置：两个维度各占一半
    color_texture_weight = 0.5
    edge_weight = 0.5
    final_score = (ct_score * color_texture_weight) + (e_score * edge_weight)

    duration = (time.perf_counter() - start_time) * 1000
    return final_score, ct_score, e_score, edges_imgs, ct_details, e_details, duration
