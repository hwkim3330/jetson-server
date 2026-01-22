"""Tesla FSD-style visualization with cyan color scheme."""

from typing import List, Tuple, Optional
import numpy as np
import cv2


# Tesla-style cyan colors (BGR format)
CYAN = (255, 212, 0)        # #00d4ff
CYAN_DIM = (255, 212, 0)    # dimmer
CYAN_GLOW = (200, 180, 0)   # glow
WHITE = (255, 255, 255)
WHITE_DIM = (180, 180, 180)


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
    path_color: Tuple[int, int, int] = CYAN,
    path_width: int = 100
) -> np.ndarray:
    """Draw Tesla FSD-style trajectory path with cyan glow."""
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

    # Fill path with semi-transparent cyan
    polygon = np.array(left_pts + right_pts[::-1], dtype=np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [polygon], (100, 60, 0))  # Dark cyan fill
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

    # Glow effect layers
    for offset, alpha in [(20, 0.05), (12, 0.08), (6, 0.1)]:
        glow_left = [(p[0] - offset, p[1]) for p in left_pts]
        glow_right = [(p[0] + offset, p[1]) for p in right_pts]
        glow_poly = np.array(glow_left + glow_right[::-1], dtype=np.int32)
        glow_overlay = img.copy()
        cv2.fillPoly(glow_overlay, [glow_poly], CYAN_GLOW)
        cv2.addWeighted(glow_overlay, alpha, img, 1 - alpha, 0, img)

    # Edge lines with gradient
    for i in range(len(left_pts) - 1):
        alpha = 1.0 - (i / len(left_pts)) * 0.6
        thickness = max(1, int(3 * alpha))
        color = tuple(int(c * alpha) for c in CYAN)
        cv2.line(img, left_pts[i], left_pts[i + 1], color, thickness, cv2.LINE_AA)
        cv2.line(img, right_pts[i], right_pts[i + 1], color, thickness, cv2.LINE_AA)

    # Center chevrons (direction markers)
    for i in range(0, len(points_2d) - 1, 4):
        pt = points_2d[i]
        alpha = 1.0 - (i / len(points_2d)) * 0.5
        size = int(15 * alpha)
        pts = np.array([
            [pt[0], pt[1] - size],
            [pt[0] - size // 2, pt[1]],
            [pt[0] + size // 2, pt[1]]
        ], dtype=np.int32)
        color = tuple(int(c * alpha) for c in WHITE)
        cv2.polylines(img, [pts], True, color, max(1, int(2 * alpha)), cv2.LINE_AA)

    return img


def draw_hud(
    img: np.ndarray,
    fps: float,
    coc_text: str,
    linear_vel: float,
    angular_vel: float,
    ai_enabled: bool
) -> np.ndarray:
    """Draw minimal Tesla-style HUD overlay."""
    h, w = img.shape[:2]

    # Top gradient bar
    for i in range(80):
        alpha = 1.0 - i / 80
        color = (int(10 * alpha), int(10 * alpha), int(10 * alpha))
        cv2.line(img, (0, i), (w, i), color, 1)

    # Bottom gradient bar
    for i in range(100):
        alpha = i / 100
        color = (int(10 * alpha), int(10 * alpha), int(10 * alpha))
        cv2.line(img, (0, h - 100 + i), (w, h - 100 + i), color, 1)

    # Logo (top left)
    cv2.putText(img, "ALPAMAYO", (24, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, CYAN, 2, cv2.LINE_AA)

    # AI status badge
    if ai_enabled:
        # Glowing AUTOPILOT badge
        badge_x = 180
        badge_y = 28
        badge_w = 110
        badge_h = 28

        # Glow
        cv2.rectangle(img, (badge_x - 2, badge_y - 2),
                      (badge_x + badge_w + 2, badge_y + badge_h + 2),
                      (150, 100, 0), -1)
        cv2.rectangle(img, (badge_x, badge_y),
                      (badge_x + badge_w, badge_y + badge_h),
                      (50, 30, 0), -1)
        cv2.rectangle(img, (badge_x, badge_y),
                      (badge_x + badge_w, badge_y + badge_h),
                      CYAN, 1)
        cv2.putText(img, "AUTOPILOT", (badge_x + 8, badge_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1, cv2.LINE_AA)
    else:
        cv2.putText(img, "MANUAL", (180, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE_DIM, 1, cv2.LINE_AA)

    # FPS (top right)
    fps_text = f"{fps:.1f} FPS"
    cv2.putText(img, fps_text, (w - 100, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1, cv2.LINE_AA)

    # Speed display (bottom left)
    speed_kmh = abs(linear_vel) * 36
    cv2.putText(img, f"{int(speed_kmh)}", (32, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, WHITE, 3, cv2.LINE_AA)
    cv2.putText(img, "KM/H", (140, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE_DIM, 1, cv2.LINE_AA)

    # Steering indicator (bottom center)
    steer_cx = w // 2
    steer_y = h - 25
    steer_width = 200

    # Track
    cv2.line(img, (steer_cx - steer_width // 2, steer_y),
             (steer_cx + steer_width // 2, steer_y),
             (50, 50, 50), 3, cv2.LINE_AA)

    # Center mark
    cv2.line(img, (steer_cx, steer_y - 8), (steer_cx, steer_y + 8),
             (80, 80, 80), 2, cv2.LINE_AA)

    # Thumb
    thumb_offset = int(angular_vel * 60)
    thumb_x = steer_cx + thumb_offset
    cv2.circle(img, (thumb_x, steer_y), 10, CYAN, -1, cv2.LINE_AA)
    cv2.circle(img, (thumb_x, steer_y), 12, (150, 100, 0), 2, cv2.LINE_AA)

    # AI reasoning text (bottom right area)
    if coc_text and ai_enabled:
        text = coc_text[:60] + "..." if len(coc_text) > 60 else coc_text
        cv2.putText(img, text, (w - 450, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE_DIM, 1, cv2.LINE_AA)

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
    """Render complete Tesla FSD-style visualization frame."""
    viz = frame.copy()
    viz = draw_trajectory(viz, trajectory)
    viz = draw_hud(viz, fps, coc_text, linear_vel, angular_vel, ai_enabled)
    return viz
