import asyncio
import random


async def handle_client_for_receiving(reader, writer):
    while True:
        try:
            # Unity'den gelen pozisyon verisini al
            data = await reader.read(1024)
            if not data:
                break
            position_data = data.decode()

            # Pozisyon verisini işleme
            process_position(position_data)

            x = random.uniform(-10, 10)  # Örnek aralık: -10 ile 10 arasında
            y = random.uniform(-10, 10)
            z = random.uniform(-10, 10)
            position = f"{x},{y},{z}"

            # Veriyi gönder
            writer.write(position.encode())
            await writer.drain()

            print("Sent position data:", position)

        except Exception as e:
            print("Exception:", e)
            break

    # Bağlantıyı kapat
    writer.close()

def process_position(position_data):
    # Pozisyon verisini işleme
    print("Received position data:", position_data)

async def main():
    host_receiving = '127.0.0.1'  # TCP sunucusunun IP adresi
    port_receiving = 8888         # Veri almak için kullanılacak TCP bağlantı noktası

    server_receiving = await asyncio.start_server(handle_client_for_receiving, host_receiving, port_receiving)

    await server_receiving.serve_forever()

# Ana fonksiyonu çalıştır
asyncio.run(main())
