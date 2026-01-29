"""Top-level wrapper to allow `python -m annotation_guidelines`.

This script forwards to the package implementation and prints the path of the
created guidelines directory.
"""
from annotation_pipeline.annotation_guidelines import AnnotationGuidelines


if __name__ == "__main__":
    out = AnnotationGuidelines().save_guidelines()
    print(out)
