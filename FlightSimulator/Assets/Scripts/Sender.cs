using System;
using System.Collections;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Threading;
using UnityEngine;
using static UnityEditor.FilePathAttribute;
using UnityEngine.UIElements;
using Palmmedia.ReportGenerator.Core.Common;
//using System.Numerics;

public class InputFromTcp
{
    public InputFromTcp()
    {
        this.thrust = 0;
        this.rollPitch = new float[] { 0f, 0f };
        this.yaw = 0;
        this.toggleFlaps = false;
    }
    public float thrust;
    public float[] rollPitch;
    public float yaw;
    public bool toggleFlaps;
}

public class Sender : MonoBehaviour
{
    private TcpClient client;
    private string serverIp = "127.0.0.1"; // Python TCP sunucusunun IP adresi
    private int port = 8888; // TCP baðlantý noktasý
    public bool connected = false;
    private Plane planeScript;
    public InputFromTcp input = new InputFromTcp();
    public bool gameNotOver = true;
    public bool responded = false;

    public event Action<Vector2> RollPitchEvent;
    public event Action<float> ThrustEvent;
    public event Action<float> YawEvent;

    Vector3 planePosition;
    Quaternion planeRotation;
    Vector3 planeEulerRotation;
    Vector3 planeVelocity;
    float planeGForce;

    public GameObject cube;
    Vector3 cubePosition;
    Quaternion cubeRotation;
    Vector3 cubeEulerRotation;
    Vector3 cubeVelocity;
    float cubeGForce;


    float reward;


    void Start()
    {
        planeScript = gameObject.GetComponent<Plane>();
        planePosition = planeScript.transform.position;
        planeRotation = planeScript.transform.rotation;
        planeEulerRotation = planeRotation.eulerAngles;
        planeVelocity = Plane.GetVelocity();
        planeGForce = Plane.GetLocalGForce();

       
        cubePosition = cube.transform.position;
        cubeRotation = cube.transform.rotation;
        cubeEulerRotation = cubeRotation.eulerAngles;
        cubeVelocity = Plane.GetVelocity();
        cubeGForce = Plane.GetLocalGForce();

        ConnectToServer();
    }

    void ConnectToServer()
    {
        try
        {
            client = new TcpClient(serverIp, port);
            connected = true;
            Debug.Log("Connected to server for sending and receiving data.");

            // Pozisyon verisi göndermek için bir döngü baþlat
            Thread t = new Thread(KeyboardReceiver);
            t.Start();
            // StartCoroutine(KeyboardReceiver());
        } catch (Exception e) {
            Debug.LogError($"Failed to connect to server: {e}");
        }
    }

    void KeyboardReceiver()
    {
        while (connected)
        {
            string keyboardData = ReceiveData();
            input = JsonUtility.FromJson<InputFromTcp>(keyboardData);
        }
    }

    void ProcessReceivedDataFromExternalKeyboardWhichWillEventuallyBeGivenFromOurDeepQLearningModel(InputFromTcp input) 
    {
        if (input.thrust != 0) {
            Debug.Log($"Yaw: {input.yaw}, Roll and Pitch: {input.rollPitch[0]} {input.rollPitch[1]}, Thrust: {input.thrust}");
            ThrustEvent.Invoke(input.thrust);
        } else if (!(input.rollPitch[0] == 0.0 && input.rollPitch[1] == 0.0)) {
            RollPitchEvent.Invoke(new Vector2(input.rollPitch[0], input.rollPitch[1]));
        } else if (input.yaw != 0) {
            YawEvent.Invoke(input.yaw);
        }
    }

    IEnumerator SendAndReceiveDataLoop()
    {
        while (connected)
        {
            // Örnek olarak pozisyon verisinin güncellendiði bir metot
            Vector3 currentPosition = transform.position;

            // Pozisyon verisini sunucuya gönder
            SendPositionData(currentPosition);

            // Sunucudan gelen veriyi al
            string receivedData = ReceiveData();

            // Alýnan veriyi iþle (örneðin, nesnenin konumunu güncelle)
            ProcessReceivedData(receivedData);

            // 1 saniye bekleyin
            yield return new WaitForSeconds(1f);
        }
    }

    void SendPositionData(Vector3 position)
    {
        try
        {
            // Pozisyon verisini oluþtur
            string positionData = $"{position.x},{position.y},{position.z}";

            // Veriyi byte dizisine dönüþtür ve sunucuya gönder
            byte[] data = Encoding.ASCII.GetBytes(positionData);
            client.GetStream().Write(data, 0, data.Length);
            Debug.Log("Sent position data: " + positionData);
        }
        catch (Exception e)
        {
            Debug.LogError($"Error while sending position data: {e}");
        }
    }

    string ReceiveData()
    {
        try
        {
          

            byte[] buffer = new byte[1024];
            byte[] jsonData;
            int bytesRead = client.GetStream().Read(buffer, 0, buffer.Length);
            string receivedData = Encoding.ASCII.GetString(buffer, 0, bytesRead);
            
            if (gameNotOver)
            {
                
                var data = new
                {
                    planePositionx = planePosition.x,
                    planePositiony = planePosition.y,
                    planePositionz = planePosition.z,
                    planeEulerRotationx = planeEulerRotation.x,
                    planeEulerRotationy = planeEulerRotation.y,
                    planeEulerRotationz = planeEulerRotation.z,
                    planeVelocity = planeVelocity.y,
                    planeGForce = planeGForce,
                    cubePositionx = cubePosition.x,
                    cubePositiony = cubePosition.y,
                    cubePositionz = cubePosition.z,
                    cubeEulerRotationx = cubeEulerRotation.x,
                    cubeEulerRotationy = cubeEulerRotation.y,
                    cubeEulerRotationz = cubeEulerRotation.z,
                    cubeVelocity = cubeVelocity.y,
                    cubeGForce = cubeGForce,
                    endGame = "CONGAME",
                    reward = reward,

                };
                string json = JsonSerializer.ToJsonString(data);
                Debug.Log("JSON Output: " + json);
                //data = Encoding.ASCII.GetBytes("CONGAME");
                jsonData = Encoding.ASCII.GetBytes(json);
            } else
            {
      
                var data = new
                {
                    planePositionx = planePosition.x,
                    planePositiony = planePosition.y,
                    planePositionz = planePosition.z,
                    planeEulerRotationx = planeEulerRotation.x,
                    planeEulerRotationy = planeEulerRotation.y,
                    planeEulerRotationz = planeEulerRotation.z,
                    planeVelocity = planeVelocity.y,
                    planeGForce = planeGForce,
                    cubePositionx = cubePosition.x,
                    cubePositiony = cubePosition.y,
                    cubePositionz = cubePosition.z,
                    cubeEulerRotationx = cubeEulerRotation.x,
                    cubeEulerRotationy = cubeEulerRotation.y,
                    cubeEulerRotationz = cubeEulerRotation.z,
                    cubeVelocity = cubeVelocity.y,
                    cubeGForce = cubeGForce,
                    endGame = "ENDGAME",
                    reward = reward,

                };
                string json = JsonUtility.ToJson(data);
                //data = Encoding.ASCII.GetBytes("ENDGAME");
                jsonData = Encoding.ASCII.GetBytes(json);
            }
            client.GetStream().Write(jsonData, 0, jsonData.Length);
            this.gameNotOver = true; // Not smart but whatever
            Debug.Log("Gönderilen JSON Boyutu: " + jsonData.Length);
            Debug.Log(receivedData);
            return receivedData;
        }
        catch (Exception e)
        {
            Debug.LogError($"Error while receiving data: {e}");
            return null;
        }
    }

    void ProcessReceivedData(string receivedData)
    {
        Debug.Log($"Received data: {receivedData}");
        // Alýnan veriyi iþle (örneðin, nesnenin konumunu güncelle)
    }

    void OnDestroy()
    {
        Disconnect();
    }

    void Disconnect()
    {
        connected = false;
        if (client != null)
        {
            client.Close();
        }
    }

    void FixedUpdate()
    {
        planeScript = gameObject.GetComponent<Plane>();
        planePosition = planeScript.transform.position;
        planeRotation = planeScript.transform.rotation;
        planeEulerRotation = planeRotation.eulerAngles;
        planeVelocity = Plane.GetVelocity();
        planeGForce = Plane.GetLocalGForce();


        cubePosition = cube.transform.position;
        cubeRotation = cube.transform.rotation;
        cubeEulerRotation = cubeRotation.eulerAngles;
        cubeVelocity = Plane.GetVelocity();
        cubeGForce = Plane.GetLocalGForce();


        reward = RewardCalculator.CalculateReward(gameObject.transform, cube.transform, cubeVelocity.y, cubeGForce);
        ProcessReceivedDataFromExternalKeyboardWhichWillEventuallyBeGivenFromOurDeepQLearningModel(input);
    }
}
