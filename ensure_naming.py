import UnityEngine

all_objects = UnityEngine.Object.FindObjectsOfType(UnityEngine.GameObject)
for go in all_objects:
    UnityEngine.Debug.Log("isGameObject")