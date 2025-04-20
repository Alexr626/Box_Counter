prompt0 = """Please count the number of boxes in this image. 
    Return ONLY the number, nothing else. 
    If you can't see any boxes, return 0."""

prompt1 = """Please count the number of boxes in this image. 
    If there is a bounding box shown in the image, count the boxes in the bounding box only.
    Return ONLY the number, nothing else. 
    If you can't see any boxes, return 0."""

# reasoning prompt
prompt2 = """Count the total number of boxes within the single bin directly in front of the camera. A bin is the smallest storage unit, identified by the barcode label on the orange beam beneath it.
Only include boxes that are inside this specific bin.
Exclude boxes that belong to neighboring bins or that are located above or below (i.e., in different shelf levels).
Ignore any boxes partially shown at the edges of the image if they do not clearly belong to this bin.
If some boxes are occluded (e.g., blocked by boxes in front or on top), infer their presence using spatial reasoning.
Use visual cues like the depth of the bin and stacking patterns.
Compare with clearly visible neighboring bins: if a neighboring bin has multiple boxes arranged along the depth axis, it is likely that this bin does too. If not, assume this bin contains only one box.
Provide your reasoning first, then output only the final count of boxes in this bin as the last number in your response"""

# reasoning prompt for sft, without requiring to include reasoning in the response
prompt3 = "Count the total number of boxes located inside the bin directly in front of the camera. A bin is defined by the barcode label on the orange beam beneath it. Only count boxes that belong to this specific bin. Exclude boxes from neighboring bins or from upper or lower shelf levels. Ignore boxes at the edges of the image if they do not clearly belong to this bin. If some boxes are likely occluded, use the depth and arrangement of boxes in neighboring bins to estimate how many are hidden. Return only the final number of boxes in this bin."
