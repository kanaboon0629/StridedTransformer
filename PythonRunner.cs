using UnityEditor.Scripting.Python;
using UnityEditor;
using UnityEngine;

public class StridedTransformer
{
    [MenuItem("MyPythonScripts/StridedTransformer")]
    static void RunStridedTransformer()
    {
        PythonRunner.RunFile($"{Application.dataPath}/StridedTransformer/develop_stridedtransformer_pose3d.py");
        //PythonRunner.RunFile($"{Application.dataPath}/StridedTransformer/ensure_naming.py");
    }
}