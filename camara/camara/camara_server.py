#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2 as cv
import numpy as np
import socket
import sys
import zlib
from cv_bridge import CvBridge
from std_msgs.msg import String  # Importa el mensaje estándar de tipo String

sys.path.append('/home/alonso/git/DeepHAZMAT')
from deep_hazmat.deep_hazmat import DeepHAZMAT  

class CamaraServer(Node):
    def __init__(self):
        super().__init__('camara_server')
        self.bridge = CvBridge()

        # Inicializar el modelo DeepHAZMAT
        net_directory = '/home/alonso/git/DeepHAZMAT/net'
        self.model = DeepHAZMAT(
            k=0,
            min_confidence=0.75,
            nms_threshold=0.3,
            segmentation_enabled=True,
            net_directory=net_directory
        )

        # Configurar el servidor socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(('127.0.0.1', 10000))
        self.server_socket.listen()
        self.get_logger().info('Servidor escuchando en el puerto 10000')

        # Crear un publicador para el tópico de nombre de HAZMAT usando String
        self.hazmat_publisher = self.create_publisher(String, 'hazmat_publisher', 10)

    def process_request(self):
        try:
            conn, addr = self.server_socket.accept()
            with conn:
                self.get_logger().info(f'Conectado por {addr}')
                
                # Recibir el tamaño de la imagen comprimida
                img_size_data = conn.recv(4)
                if not img_size_data:
                    self.get_logger().error("Error: No se recibió el tamaño de la imagen.")
                    return

                img_size = int.from_bytes(img_size_data, byteorder='little')
                self.get_logger().info(f'Tamaño de la imagen recibido: {img_size} bytes')

                b_imagen = bytearray()
                while len(b_imagen) < img_size:
                    data = conn.recv(4096)
                    if not data:
                        self.get_logger().error("Error: Se interrumpió la recepción de datos.")
                        return
                    b_imagen += data

                if len(b_imagen) == 0:
                    self.get_logger().error("Error: La imagen recibida está vacía.")
                    return

                if len(b_imagen) != img_size:
                    self.get_logger().error(f"Error: Tamaño de la imagen recibido ({len(b_imagen)}) no coincide con el tamaño esperado ({img_size}).")
                    return

                # Descomprimir y decodificar la imagen
                img_enc = zlib.decompress(b_imagen)
                img_dec = np.ndarray(shape=(len(img_enc),), dtype='uint8', buffer=img_enc)
                color = cv.imdecode(img_dec, cv.IMREAD_COLOR)

                if color is None:
                    self.get_logger().error("Error: No se pudo decodificar la imagen.")
                    return

                # Procesar detecciones de HAZMAT y publicar solo el nombre
                for hazmat in self.model.update(color):
                    hazmat.draw(color, padding=0.1)

                    # Publicar el nombre del HAZMAT detectado usando el atributo `name`
                    msg = String()
                    msg.data = hazmat.name  # Usamos el atributo `name` para obtener el nombre del HAZMAT
                    self.hazmat_publisher.publish(msg)
                    self.get_logger().info(f'HAZMAT detectado publicado: {msg.data}')

                # Codificar y enviar la imagen procesada de vuelta al cliente
                _, img_encoded = cv.imencode('.jpg', color)
                conn.sendall(len(img_encoded).to_bytes(4, byteorder='little'))
                conn.sendall(img_encoded.tobytes())

        except Exception as e:
            self.get_logger().error(f'Error en el procesamiento de la solicitud: {e}')

def main(args=None):
    rclpy.init(args=args)
    camara_server = CamaraServer()

    try:
        while rclpy.ok():
            camara_server.process_request()
    except KeyboardInterrupt:
        pass
    finally:
        camara_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

