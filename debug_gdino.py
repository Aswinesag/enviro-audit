import sys
import traceback

print(f"Python: {sys.version}")
print("Attempting to import groundingdino...")

try:
    import groundingdino
    print(f"GroundingDINO package found at: {groundingdino.__file__}")
    
    from groundingdino.util.inference import load_model, load_image, predict, annotate
    print("SUCCESS: groundingdino.util.inference imported.")
except ImportError as e:
    print(f"\nIMPORT ERROR: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"\nUNKNOWN ERROR: {e}")
    traceback.print_exc()
