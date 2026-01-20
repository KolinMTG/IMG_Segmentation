"""
Post-processing pipeline for semantic segmentation masks.

This module implements a complete post-processing pipeline for 512x512 
semantic segmentation masks in satellite/aerial imagery.

Processing order:
    1. Roads (highest priority) - AGGRESSIVE topology and width regularization
    2. Buildings - geometry regularization and expansion
    3. Natural classes - noise removal and smoothing
    4. Conflict resolution - enforce class hierarchy

Class priority: Building > Road > Water > Woodland > Field

AGGRESSIVE ROAD PROCESSING:
    - Long-distance road linking across image
    - Road extrapolation to borders
    - Global continuity enforcement
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor, LinearRegression
from typing import Tuple, Optional, List
import logging

log = logging.getLogger(__name__)


def posttreat_pipeline(
    mask: np.ndarray,
    **kwargs
) -> np.ndarray:
    """
    Main post-processing pipeline for semantic segmentation masks.
    
    Args:
        mask: 2D numpy array of shape (512, 512) with integer class IDs
        **kwargs: Optional parameters for fine-tuning
            - road_width_percentile: Percentile for road width estimation (default: 75)
            - road_connection_distance: Max distance to connect road fragments (default: 25)
            - road_extension_enabled: Enable aggressive road extension (default: True)
            - road_bridging_enabled: Enable aggressive road bridging (default: True)
            - building_min_area: Minimum area for buildings (default: 50)
            - building_expansion_pixels: Pixels to expand buildings (default: 2)
            - natural_min_area: Minimum area for natural classes (default: 100)
            - morphology_kernel_size: Kernel size for morphological ops (default: 5)
    
    Returns:
        Post-processed mask of shape (512, 512) with clean class assignments
    """
    log.info("Starting post-processing pipeline with AGGRESSIVE road processing")
    
    #! Extract class IDs (assumes external definition, e.g., from cste.py)
    class_field = kwargs.get('class_field', 0)
    class_building = kwargs.get('class_building', 1)
    class_woodland = kwargs.get('class_woodland', 2)
    class_water = kwargs.get('class_water', 3)
    class_road = kwargs.get('class_road', 4)
    
    # Create working copy
    processed_mask = mask.copy()
    
    #! Step 1: AGGRESSIVE road processing (topology, extension, bridging)
    log.info("Processing roads: AGGRESSIVE topology and continuity enforcement")
    processed_mask = _process_roads_aggressive(
        processed_mask,
        class_road,
        class_building,
        width_percentile=kwargs.get('road_width_percentile', 75),
        connection_distance=kwargs.get('road_connection_distance', 25),
        min_length=kwargs.get('road_min_length', 20),
        enable_extension=kwargs.get('road_extension_enabled', True),
        enable_bridging=kwargs.get('road_bridging_enabled', True),
        min_component_size=kwargs.get('road_min_component_size', 100)
    )
    
    #! Step 2: Process buildings (geometry regularization)
    log.info("Processing buildings: regularization and expansion")
    processed_mask = _process_buildings(
        processed_mask,
        class_building,
        min_area=kwargs.get('building_min_area', 50),
        expansion_pixels=kwargs.get('building_expansion_pixels', 2),
        regularization_threshold=kwargs.get('building_regularization_threshold', 0.85)
    )
    
    #! Step 3: Smooth natural classes (remove noise)
    log.info("Smoothing natural classes: field, woodland, water")
    processed_mask = _smooth_natural_classes(
        processed_mask,
        class_field,
        class_woodland,
        class_water,
        min_area=kwargs.get('natural_min_area', 100),
        kernel_size=kwargs.get('morphology_kernel_size', 5)
    )
    
    #! Step 4: Resolve conflicts using class hierarchy
    log.info("Resolving conflicts with class priority")
    processed_mask = _resolve_conflicts(
        processed_mask,
        class_building,
        class_road,
        class_water,
        class_woodland,
        class_field
    )
    
    log.info("Post-processing pipeline completed")
    return processed_mask


def _process_roads_aggressive(
    mask: np.ndarray,
    class_road: int,
    class_building: int,
    width_percentile: int = 75,
    connection_distance: int = 25,
    min_length: int = 20,
    enable_extension: bool = True,
    enable_bridging: bool = True,
    min_component_size: int = 100
) -> np.ndarray:
    """
    AGGRESSIVE road processing: enforce global continuity and linearity.
    
    Strategy:
        1. Extract road binary mask and estimate target width
        2. Identify major road components
        3. AGGRESSIVE: Bridge aligned road components across gaps
        4. AGGRESSIVE: Extend roads to image borders
        5. Skeletonize and reconstruct with uniform width
        
    Args:
        mask: Current segmentation mask
        class_road: Road class ID
        class_building: Building class ID (protected from override)
        width_percentile: Percentile for width estimation
        connection_distance: Max pixel distance to connect fragments
        min_length: Minimum road segment length to preserve
        enable_extension: Enable road extension to borders
        enable_bridging: Enable road bridging across gaps
        min_component_size: Minimum size for major road components
        
    Returns:
        Mask with aggressively processed roads
    """
    road_mask = (mask == class_road).astype(np.uint8)
    building_mask = (mask == class_building).astype(np.uint8)
    
    if road_mask.sum() == 0:
        log.info("No roads detected, skipping road processing")
        return mask
    
    #! Estimate target road width from existing large components
    labeled_roads = label(road_mask, connectivity=2)
    regions = regionprops(labeled_roads)
    
    widths = []
    major_components = []
    
    for region in regions:
        if region.area > 50:
            if region.major_axis_length > 0:
                estimated_width = region.area / region.major_axis_length
                widths.append(estimated_width)
            
            # Track major road components for aggressive processing
            if region.area >= min_component_size:
                major_components.append(region)
    
    if len(widths) > 0:
        target_width = int(np.percentile(widths, width_percentile))
        target_width = max(3, min(target_width, 12))
    else:
        target_width = 5
    
    log.info(f"Target road width: {target_width} pixels")
    log.info(f"Found {len(major_components)} major road components")
    
    #! Remove very small noise
    road_mask_clean = remove_small_objects(road_mask.astype(bool), min_size=min_length, connectivity=2).astype(np.uint8)
    
    #! AGGRESSIVE: Bridge aligned road components
    if enable_bridging and len(major_components) >= 2:
        log.info("AGGRESSIVE: Bridging aligned road components")
        road_mask_clean = _bridge_road_components(
            road_mask_clean,
            major_components,
            mask,
            class_building,
            target_width,
            max_gap=100,
            angle_tolerance=25
        )
    
    #! AGGRESSIVE: Extend roads to borders
    if enable_extension and len(major_components) > 0:
        log.info("AGGRESSIVE: Extending roads to image borders")
        road_mask_clean = _extend_roads_to_borders(
            road_mask_clean,
            major_components,
            mask,
            class_building,
            target_width,
            min_straight_length=50,
            border_proximity=30
        )
    
    #! Skeletonize to get road centerlines
    skeleton = skeletonize(road_mask_clean > 0).astype(np.uint8)
    
    #! Connect nearby skeleton endpoints
    skeleton_connected = _connect_road_fragments(skeleton, connection_distance)
    
    #! Dilate skeleton to target width
    dilation_radius = target_width // 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_radius * 2 + 1, dilation_radius * 2 + 1))
    road_reconstructed = cv2.dilate(skeleton_connected, kernel, iterations=1)
    
    #! Update mask with processed roads (preserve buildings)
    result_mask = mask.copy()
    road_pixels = (road_reconstructed > 0) & (building_mask == 0)
    result_mask[road_pixels] = class_road
    
    return result_mask


def _bridge_road_components(
    road_mask: np.ndarray,
    major_components: List,
    original_mask: np.ndarray,
    class_building: int,
    target_width: int,
    max_gap: int = 100,
    angle_tolerance: float = 25
) -> np.ndarray:
    """
    AGGRESSIVE: Bridge aligned road components across gaps.
    
    Args:
        road_mask: Current road binary mask
        major_components: List of major road region properties
        original_mask: Original segmentation mask
        class_building: Building class ID (protected)
        target_width: Target road width
        max_gap: Maximum gap distance to bridge
        angle_tolerance: Angle tolerance in degrees for alignment
        
    Returns:
        Road mask with bridged components
    """
    result_mask = road_mask.copy()
    h, w = road_mask.shape
    building_mask = (original_mask == class_building).astype(np.uint8)
    
    #! AGGRESSIVE: Check all pairs of major components
    for i, comp1 in enumerate(major_components):
        for comp2 in major_components[i + 1:]:
            
            # Get component centroids
            y1, x1 = comp1.centroid
            y2, x2 = comp2.centroid
            
            # Calculate distance
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            if distance > max_gap or distance < 10:
                continue
            
            #! AGGRESSIVE: Calculate dominant orientations
            angle1 = comp1.orientation * 180 / np.pi
            angle2 = comp2.orientation * 180 / np.pi
            
            # Normalize angles to [-90, 90]
            angle1 = ((angle1 + 90) % 180) - 90
            angle2 = ((angle2 + 90) % 180) - 90
            
            angle_diff = abs(angle1 - angle2)
            if angle_diff > 90:
                angle_diff = 180 - angle_diff
            
            #! AGGRESSIVE: If aligned, bridge the gap
            if angle_diff <= angle_tolerance:
                log.info(f"Bridging components: distance={distance:.1f}px, angle_diff={angle_diff:.1f}°")
                
                # Draw thick line between centroids
                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))
                
                thickness = max(3, target_width)
                
                # Create bridge line avoiding buildings
                bridge_mask = np.zeros_like(result_mask)
                cv2.line(bridge_mask, pt1, pt2, 1, thickness=thickness)
                
                # Only add bridge where no buildings exist
                bridge_pixels = (bridge_mask > 0) & (building_mask == 0)
                result_mask[bridge_pixels] = 1
    
    return result_mask


def _extend_roads_to_borders(
    road_mask: np.ndarray,
    major_components: List,
    original_mask: np.ndarray,
    class_building: int,
    target_width: int,
    min_straight_length: int = 50,
    border_proximity: int = 30
) -> np.ndarray:
    """
    AGGRESSIVE: Extend straight road segments to image borders.
    
    Args:
        road_mask: Current road binary mask
        major_components: List of major road region properties
        original_mask: Original segmentation mask
        class_building: Building class ID (protected)
        target_width: Target road width
        min_straight_length: Minimum length to consider for extension
        border_proximity: Distance from border to trigger extension
        
    Returns:
        Road mask with extended roads
    """
    result_mask = road_mask.copy()
    h, w = road_mask.shape
    building_mask = (original_mask == class_building).astype(np.uint8)
    
    for comp in major_components:
        
        # Only extend sufficiently long components
        if comp.major_axis_length < min_straight_length:
            continue
        
        #! AGGRESSIVE: Check if component is near border
        minr, minc, maxr, maxc = comp.bbox
        
        near_top = minr < border_proximity
        near_bottom = maxr > (h - border_proximity)
        near_left = minc < border_proximity
        near_right = maxc > (w - border_proximity)
        
        if not (near_top or near_bottom or near_left or near_right):
            continue
        
        #! AGGRESSIVE: Fit line to component using PCA
        coords = np.argwhere(result_mask[minr:maxr, minc:maxc] > 0)
        
        if len(coords) < 10:
            continue
        
        # Shift coords to global coordinates
        coords[:, 0] += minr
        coords[:, 1] += minc
        
        # Fit line using PCA for robust direction
        pca = PCA(n_components=1)
        try:
            pca.fit(coords)
            direction = pca.components_[0]
            centroid = coords.mean(axis=0)
        except:
            continue
        
        # Normalize direction
        direction = direction / np.linalg.norm(direction)
        
        #! AGGRESSIVE: Extend in both directions to borders
        log.info(f"Extending road component at ({int(centroid[0])}, {int(centroid[1])})")
        
        for sign in [-1, 1]:
            # Extend along direction
            extension_dir = sign * direction
            
            # Calculate how far to extend to reach border
            max_extension = max(h, w)
            
            for step in range(0, max_extension, 2):
                point = centroid + step * extension_dir
                y, x = int(point[0]), int(point[1])
                
                # Stop if out of bounds
                if y < 0 or y >= h or x < 0 or x >= w:
                    break
                
                # Draw road extension with target width
                thickness = max(3, target_width)
                
                # Create extension segment
                start_point = (int(centroid[1]), int(centroid[0]))
                end_point = (x, y)
                
                extension_mask = np.zeros_like(result_mask)
                cv2.line(extension_mask, start_point, end_point, 1, thickness=thickness)
                
                # Only add where no buildings exist
                extension_pixels = (extension_mask > 0) & (building_mask == 0)
                result_mask[extension_pixels] = 1
    
    return result_mask


def _detect_long_road_axes(
    road_mask: np.ndarray,
    min_length: int = 100
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Detect long, straight road axes using Hough line transform.
    
    Args:
        road_mask: Binary road mask
        min_length: Minimum line length to detect
        
    Returns:
        List of (start_point, end_point, angle) tuples
    """
    edges = cv2.Canny(road_mask * 255, 50, 150)
    
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min_length,
        maxLineGap=20
    )
    
    road_axes = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle
            dx = x2 - x1
            dy = y2 - y1
            angle = np.arctan2(dy, dx) * 180 / np.pi
            
            start_point = np.array([y1, x1])
            end_point = np.array([y2, x2])
            
            road_axes.append((start_point, end_point, angle))
    
    return road_axes


def _connect_road_fragments(
    skeleton: np.ndarray,
    max_distance: int = 25
) -> np.ndarray:
    """
    Connect nearby road skeleton endpoints to improve topology.
    
    Args:
        skeleton: Binary skeleton image
        max_distance: Maximum distance to connect endpoints
        
    Returns:
        Connected skeleton
    """
    skeleton_connected = skeleton.copy()
    
    #! Find endpoints (pixels with exactly one neighbor)
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    endpoints = np.argwhere((skeleton > 0) & (neighbor_count == 1))
    
    if len(endpoints) < 2:
        return skeleton_connected
    
    #! Connect close endpoints
    connected_pairs = set()
    
    for i, pt1 in enumerate(endpoints):
        for j, pt2 in enumerate(endpoints[i + 1:], start=i + 1):
            distance = np.linalg.norm(pt1 - pt2)
            
            if distance > 0 and distance <= max_distance:
                # Avoid duplicate connections
                pair = tuple(sorted([i, j]))
                if pair not in connected_pairs:
                    connected_pairs.add(pair)
                    
                    # Draw line between endpoints
                    cv2.line(
                        skeleton_connected,
                        (pt1[1], pt1[0]),
                        (pt2[1], pt2[0]),
                        1,
                        thickness=1
                    )
    
    return skeleton_connected


def _process_buildings(
    mask: np.ndarray,
    class_building: int,
    min_area: int = 50,
    expansion_pixels: int = 2,
    regularization_threshold: float = 0.85
) -> np.ndarray:
    """
    Process building class: regularize geometry, expand slightly.
    
    Strategy:
        1. Extract building components
        2. Filter by minimum area
        3. Regularize each building to rectangular shape
        4. Expand buildings slightly (never shrink)
        
    Args:
        mask: Current segmentation mask
        class_building: Building class ID
        min_area: Minimum building area to preserve
        expansion_pixels: Pixels to expand building boundaries
        regularization_threshold: Threshold for rectangular fit quality
        
    Returns:
        Mask with processed buildings
    """
    building_mask = (mask == class_building).astype(np.uint8)
    
    if building_mask.sum() == 0:
        log.info("No buildings detected, skipping building processing")
        return mask
    
    #! Extract connected components
    labeled_buildings = label(building_mask, connectivity=2)
    regions = regionprops(labeled_buildings)
    
    regularized_mask = np.zeros_like(building_mask)
    
    for region in regions:
        if region.area < min_area:
            continue
        
        #! Regularize to rectangle (axis-aligned or rotated)
        minr, minc, maxr, maxc = region.bbox
        
        # Get oriented bounding box
        contours, _ = cv2.findContours(
            (labeled_buildings == region.label).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) > 0:
            contour = contours[0]
            
            # Try axis-aligned rectangle first
            x, y, w, h = cv2.boundingRect(contour)
            axis_aligned_area = w * h
            
            # Try rotated rectangle
            if len(contour) >= 5:
                rotated_rect = cv2.minAreaRect(contour)
                rotated_area = rotated_rect[1][0] * rotated_rect[1][1]
                
                # Use rotated if significantly better fit
                if rotated_area > 0 and region.area / rotated_area > regularization_threshold:
                    box = cv2.boxPoints(rotated_rect)
                    box = np.round(box).astype(np.int32)
                    cv2.fillPoly(regularized_mask, [box], 1)
                else:
                    # Use axis-aligned
                    cv2.rectangle(regularized_mask, (x, y), (x + w, y + h), 1, -1)
            else:
                # Fallback to axis-aligned
                cv2.rectangle(regularized_mask, (x, y), (x + w, y + h), 1, -1)
    
    #! Expand buildings slightly
    if expansion_pixels > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expansion_pixels * 2 + 1, expansion_pixels * 2 + 1))
        regularized_mask = cv2.dilate(regularized_mask, kernel, iterations=1)
    
    #! Update mask
    result_mask = mask.copy()
    result_mask[regularized_mask > 0] = class_building
    
    return result_mask


def _smooth_natural_classes(
    mask: np.ndarray,
    class_field: int,
    class_woodland: int,
    class_water: int,
    min_area: int = 100,
    kernel_size: int = 5
) -> np.ndarray:
    """
    Smooth natural classes: remove small regions, resolve local conflicts.
    
    Strategy:
        1. For each natural class, remove small isolated regions
        2. Reassign removed pixels to dominant neighboring class
        3. Apply light morphological smoothing
        
    Args:
        mask: Current segmentation mask
        class_field: Field class ID
        class_woodland: Woodland class ID
        class_water: Water class ID
        min_area: Minimum region area to preserve
        kernel_size: Kernel size for morphological smoothing
        
    Returns:
        Mask with smoothed natural classes
    """
    result_mask = mask.copy()
    natural_classes = [class_field, class_woodland, class_water]
    
    for cls in natural_classes:
        class_mask = (result_mask == cls).astype(np.uint8)
        
        if class_mask.sum() == 0:
            continue
        
        #! Remove small noisy regions
        labeled_class = label(class_mask, connectivity=2)
        regions = regionprops(labeled_class)
        
        for region in regions:
            if region.area < min_area:
                # Get bounding box
                minr, minc, maxr, maxc = region.bbox
                pad = 3
                minr_pad = max(0, minr - pad)
                minc_pad = max(0, minc - pad)
                maxr_pad = min(result_mask.shape[0], maxr + pad)
                maxc_pad = min(result_mask.shape[1], maxc + pad)
                
                # Find dominant neighboring class
                neighborhood = result_mask[minr_pad:maxr_pad, minc_pad:maxc_pad]
                region_pixels = (labeled_class == region.label)
                
                # Count neighboring classes (excluding current class)
                neighbor_classes = neighborhood[~region_pixels[minr_pad:maxr_pad, minc_pad:maxc_pad]]
                if len(neighbor_classes) > 0:
                    unique, counts = np.unique(neighbor_classes, return_counts=True)
                    # Exclude current class
                    valid_idx = unique != cls
                    if valid_idx.sum() > 0:
                        dominant_class = unique[valid_idx][np.argmax(counts[valid_idx])]
                        result_mask[region_pixels] = dominant_class
        
        #! Light morphological smoothing (closing then opening)
        class_mask_smooth = (result_mask == cls).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        class_mask_smooth = cv2.morphologyEx(class_mask_smooth, cv2.MORPH_CLOSE, kernel)
        class_mask_smooth = cv2.morphologyEx(class_mask_smooth, cv2.MORPH_OPEN, kernel)
        
        result_mask[class_mask_smooth > 0] = cls
    
    return result_mask


def _resolve_conflicts(
    mask: np.ndarray,
    class_building: int,
    class_road: int,
    class_water: int,
    class_woodland: int,
    class_field: int
) -> np.ndarray:
    """
    Resolve overlapping classes using priority hierarchy.
    
    Priority: Building > Road > Water > Woodland > Field
    
    Strategy:
        1. Detect multi-class pixels (shouldn't exist, but handle gracefully)
        2. Assign to highest priority class
        3. Fill any remaining unlabeled pixels with field (background)
        
    Args:
        mask: Current segmentation mask
        class_building: Building class ID (highest priority)
        class_road: Road class ID
        class_water: Water class ID
        class_woodland: Woodland class ID
        class_field: Field class ID (lowest priority, background)
        
    Returns:
        Clean mask with no conflicts
    """
    result_mask = mask.copy()
    
    #! Enforce class hierarchy (highest to lowest priority)
    priority_order = [
        class_building,
        class_road,
        class_water,
        class_woodland,
        class_field
    ]
    
    # Create clean mask by applying classes in priority order
    clean_mask = np.full_like(result_mask, class_field)
    
    for cls in reversed(priority_order):
        class_mask = (result_mask == cls)
        clean_mask[class_mask] = cls
    
    #! Final sanity check: ensure all pixels are labeled
    valid_classes = set(priority_order)
    invalid_pixels = ~np.isin(clean_mask, list(valid_classes))
    
    if invalid_pixels.sum() > 0:
        log.info(f"Reassigning {invalid_pixels.sum()} invalid pixels to field class")
        clean_mask[invalid_pixels] = class_field
    
    return clean_mask