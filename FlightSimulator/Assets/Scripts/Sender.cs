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
    public InputFromTcp input = new InputFromTcp();
    public bool gameNotOver = true;
    public bool responded = false;

    public event Action<Vector2> RollPitchEvent;
    public event Action<float> ThrustEvent;
    public event Action<float> YawEvent;

    private GameObject playerPlane;
    private Plane playerPlaneScript;
    Vector3 playerPlanePosition;
    Vector3 playerPlaneEulerRotation;
    Vector3 playerPlaneVelocity;
    float playerPlaneGForce;

    private GameObject enemyPlane;
    private Plane enemyPlaneScript;
    Vector3 enemyPlanePosition;
    Vector3 enemyPlaneEulerRotation;
    Vector3 enemyPlaneVelocity;
    float enemyPlaneGForce;
    float reward;


    void Start()
    {
        playerPlane = GameObject.FindWithTag("PlayerPlane");
        enemyPlane = GameObject.FindWithTag("EnemyPlane");
        playerPlaneScript = playerPlane.GetComponent<Plane>();
        enemyPlaneScript = enemyPlane.GetComponent<Plane>();

        playerPlanePosition = playerPlane.transform.position;
        playerPlaneEulerRotation = playerPlane.transform.rotation.eulerAngles;
        playerPlaneVelocity = playerPlaneScript.GetVelocity();
        playerPlaneGForce = playerPlaneScript.GetLocalGForce();

        enemyPlanePosition = enemyPlane.transform.position;
        enemyPlaneEulerRotation = enemyPlane.transform.rotation.eulerAngles;
        enemyPlaneVelocity = enemyPlaneScript.GetVelocity();
        enemyPlaneGForce = enemyPlaneScript.GetLocalGForce();

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
            Debug.Log(keyboardData);
            this.input = JsonUtility.FromJson<InputFromTcp>(keyboardData);
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
                    playerPlanePositionx = playerPlanePosition.x,
                    playerPlanePositiony = playerPlanePosition.y,
                    playerPlanePositionz = playerPlanePosition.z,
                    playerPlaneEulerRotationx = playerPlaneEulerRotation.x,
                    playerPlaneEulerRotationy = playerPlaneEulerRotation.y,
                    playerPlaneEulerRotationz = playerPlaneEulerRotation.z,
                    playerPlaneForwardVelocity = playerPlaneVelocity.z,
                    playerPlaneGForce = playerPlaneGForce,
                    enemyPlanePositionx = enemyPlanePosition.x,
                    enemyPlanePositiony = enemyPlanePosition.y,
                    enemyPlanePositionz = enemyPlanePosition.z,
                    enemyPlaneEulerRotationx = enemyPlaneEulerRotation.x,
                    enemyPlaneEulerRotationy = enemyPlaneEulerRotation.y,
                    enemyPlaneEulerRotationz = enemyPlaneEulerRotation.z,
                    enemyPlaneForwardVelocity = enemyPlaneVelocity.z,
                    enemyPlaneGForce = enemyPlaneGForce,
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
                    playerPlanePositionx = playerPlanePosition.x,
                    playerPlanePositiony = playerPlanePosition.y,
                    playerPlanePositionz = playerPlanePosition.z,
                    playerPlaneEulerRotationx = playerPlaneEulerRotation.x,
                    playerPlaneEulerRotationy = playerPlaneEulerRotation.y,
                    playerPlaneEulerRotationz = playerPlaneEulerRotation.z,
                    playerPlaneForwardVelocity = playerPlaneVelocity.z,
                    playerPlaneGForce = playerPlaneGForce,
                    enemyPlanePositionx = enemyPlanePosition.x,
                    enemyPlanePositiony = enemyPlanePosition.y,
                    enemyPlanePositionz = enemyPlanePosition.z,
                    enemyPlaneEulerRotationx = enemyPlaneEulerRotation.x,
                    enemyPlaneEulerRotationy = enemyPlaneEulerRotation.y,
                    enemyPlaneEulerRotationz = enemyPlaneEulerRotation.z,
                    enemyPlaneForwardVelocity = enemyPlaneVelocity.z,
                    enemyPlaneGForce = enemyPlaneGForce,
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
        playerPlanePosition = playerPlane.transform.position;
        playerPlaneEulerRotation = playerPlane.transform.rotation.eulerAngles;
        playerPlaneVelocity = playerPlaneScript.GetVelocity();
        playerPlaneGForce = playerPlaneScript.GetLocalGForce();


        enemyPlanePosition = enemyPlane.transform.position;
        enemyPlaneEulerRotation = enemyPlane.transform.rotation.eulerAngles;
        enemyPlaneVelocity = enemyPlaneScript.GetVelocity();
        enemyPlaneGForce = enemyPlaneScript.GetLocalGForce();


        reward = RewardCalculator.CalculateReward(playerPlane.transform, enemyPlane.transform, enemyPlaneVelocity.z, enemyPlaneGForce);
        ProcessReceivedDataFromExternalKeyboardWhichWillEventuallyBeGivenFromOurDeepQLearningModel(this.input);
    }
}
