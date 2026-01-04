

def mask_to_labels_correlation(predicted_mask, label_mask):
    """
    Compute correlation between predicted mask and label mask.
    
    Args:
        predicted_mask: 2D numpy array of shape (H, W) with predicted class IDs
        label_mask: 2D numpy array of shape (H, W) with true class IDs"""

def extract_binary_mask(label_mask, class_id):
    """
    Extract a binary mask for a specific class from the label mask.
    
    Args:
        label_mask: 2D numpy array of shape (H, W) with class IDs
        class_id: Class ID to extract
    
    Returns:
        binary_mask: 2D numpy array of shape (H, W) with 1 for class_id and 0 elsewhere
    """