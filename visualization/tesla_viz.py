"""Tesla FSD-style visualization."""

from typing import List, Tuple, Optional
import numpy as np
import cv2


def project_trajectory(
    traj_3d: np.ndarray,
    img_w: int,
    img_h: int,
    fx: float = 500,
    camera_height: float = 0.35
) -> List[Tuple[int, int]]:
    """Project 3D trajectory to 2D image coordinates."""
    cx = img_w // 2
    cy = img_h + 50

    points_2d = []
    for p in traj_3d:
        x_forward = p[0]
        y_left = p[1]

        if x_forward < 0.01:
            continue

        u = cx - (y_left / x_forward) * fx
        v = cy - (camera_height / x_forward) * fx

        if -50 <= u < img_w + 50 and 0 <= v < img_h:
            points_2d.append((int(u), int(v)))

    return points_2d


def draw_trajectory(
    img: np.ndarray,
    traj_3d: Optional[np.ndarray],
    path_color: Tuple[int, int, int] = (180, 230, 50),
    path_width: int = 100
) -> np.ndarray:
    """Draw Tesla FSD-style trajectory path."""
    if traj_3d is None:
        return img

    h, w = img.shape[:2]
    points_2d = project_trajectory(traj_3d, w, h)

    if len(points_2d) < 3:
        return img

    # Build path polygon
    left_pts = []
    right_pts = []

    for pt in points_2d:
        y_ratio = pt[1] / h
        width = int(path_width * (0.1 + 0.9 * y_ratio))
        left_pts.append((pt[0] - width // 2, pt[1]))
        right_pts.append((pt[0] + width // 2, pt[1]))

    if len(left_pts) < 2:
        return img

    # Fill path
    polygon = np.array(left_pts + right_pts[::-1], dtype=np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [polygon], path_color)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

    # Glow effect
    edge_color = (200, 255, 100)
    for offset in [15, 10, 5]:
        glow_left = [(p[0] - offset, p[1]) for p in left_pts]
        glow_right = [(p[0] + offset, p[1]) for p in right_pts]
        glow_poly = np.array(glow_left + glow_right[::-1], dtype=np.int32)
        glow_overlay = img.copy()
        cv2.fillPoly(glow_overlay, [glow_poly], edge_color)
        cv2.addWeighted(glow_overlay, 0.08, img, 0.92, 0, img)

    # Edge lines
    for i in range(len(left_pts) - 1):
        alpha = 1.0 - (i / len(left_pts)) * 0.7
        thickness = max(1, int(3 * alpha))
        color = tuple(int(c * alpha) for c in edge_color)
        cv2.line(img, left_pts[i], left_pts[i + 1], color, thickness, cv2.LINE_AA)
        cv2.line(img, right_pts[i], right_pts[i + 1], color, thickness, cv2.LINE_AA)

    # Center chevrons
    for i in range(0, len(points_2d) - 1, 5):
        pt = points_2d[i]
        alpha = 1.0 - (i / len(points_2d)) * 0.5
        size = int(12 * alpha)
        pts = np.array([
            [pt[0], pt[1] - size],
            [pt[0] - size // 2, pt[1]],
            [pt[0] + size // 2, pt[1]]
        ], dtype=np.int32)
        cv2.polylines(img, [pts], True, (255, 255, 255), max(1, int(2 * alpha)), cv2.LINE_AA)

    return img


def draw_hud(
    img: np.ndarray,
    fps: float,
    coc_text: str,
    linear_vel: float,
    angular_vel: float,
    ai_enabled: bool
) -> np.ndarray:
    """Draw Tesla-style HUD overlay."""
    h, w = img.shape[:2]

    # Top gradient
    for i in range(60):
        alpha = 1.0 - i / 60
        color = (int(15 * alpha), int(15 * alpha), int(15 * alpha))
        cv2.line(img, (0, i), (w, i), color, 1)

    # Logo
    cv2.putText(img, "ALPAMAYO", (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

    # AI status
    if ai_enabled:
        cv2.circle(img, (w - 30, 30), 12, (50, 255, 50), -1)
        cv2.circle(img, (w - 30, 30), 12, (100, 255, 100), 2)
        cv2.putText(img, "AI", (w - 38, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 50, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "AUTOPILOT", (w - 140, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2, cv2.LINE_AA)
    else:
        cv2.circle(img, (w - 30, 30), 12, (80, 80, 80), -1)
        cv2.putText(img, "MANUAL", (w - 120, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)

    # Speed display
    speed_kmh = abs(linear_vel) * 36  # scaled
    cv2.putText(img, f"{int(speed_kmh)}", (30, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, "km/h", (130, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)

    # FPS
    cv2.putText(img, f"{fps:.1f} FPS", (w - 100, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

    # CoC text bar
    if coc_text:
        for i in range(70):
            alpha = i / 70
            color = (int(20 * alpha), int(20 * alpha), int(20 * alpha))
            cv2.line(img, (0, h - 70 + i), (w, h - 70 + i), color, 1)

        cv2.putText(img, "AI:", (200, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(img, coc_text[:70], (235, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # Steering indicator
    steer_x = w // 2
    steer_y = h - 15
    steer_offset = int(angular_vel * 50)
    cv2.line(img, (steer_x - 150, steer_y), (steer_x + 150, steer_y), (60, 60, 60), 4, cv2.LINE_AA)
    cv2.circle(img, (steer_x + steer_offset, steer_y), 8, (50, 255, 50), -1)

    return img


def render_frame(
    frame: np.ndarray,
    trajectory: Optional[np.ndarray],
    fps: float,
    coc_text: str,
    linear_vel: float,
    angular_vel: float,
    ai_enabled: bool
) -> np.ndarray:
    """Render complete visualization frame."""
    viz = frame.copy()
    viz = draw_trajectory(viz, trajectory)
    viz = draw_hud(viz, fps, coc_text, linear_vel, angular_vel, ai_enabled)
    return viz
