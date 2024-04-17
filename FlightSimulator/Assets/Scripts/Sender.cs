using System;
using System.Collections;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

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

    void Start()
    {
        planeScript = gameObject.GetComponent<Plane>();
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
            byte[] data;
            int bytesRead = client.GetStream().Read(buffer, 0, buffer.Length);
            string receivedData = Encoding.ASCII.GetString(buffer, 0, bytesRead);
            if (gameNotOver)
            {
                data = Encoding.ASCII.GetBytes("CONGAME");
            } else
            {
                data = Encoding.ASCII.GetBytes("ENDGAME");
            }
            client.GetStream().Write(data, 0, data.Length);
            this.gameNotOver = true; // Not smart but whatever
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
        ProcessReceivedDataFromExternalKeyboardWhichWillEventuallyBeGivenFromOurDeepQLearningModel(input);
    }
}
