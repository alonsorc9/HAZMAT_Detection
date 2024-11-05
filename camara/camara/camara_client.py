#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2 as cv
import numpy as np
import socket
from cv_bridge import CvBridge
import zlib 

class CamaraClient(Node):
    def __init__(self):
        super().__init__('camara_client')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.image_callback,
            10
        )
        
        self.server_address = ('127.0.0.1', 10000)

    def image_callback(self, msg):
        try:
            imagen = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Error al convertir la imagen: {e}")
            return

        if imagen is None or imagen.size == 0:
            self.get_logger().error("Error: La imagen está vacía después de la conversión.")
            return
        
        # Reducir la resolución de la imagen
        imagen = cv.resize(imagen, (320, 240))  # Ajustar para menor tiempo de procesamiento

        # Codificar la imagen como JPEG
        result, img_enc = cv.imencode('.jpg', imagen)
        if not result:
            self.get_logger().error("Error al codificar la imagen.")
            return
        
        # Comprimir la imagen codificada
        img_compressed = zlib.compress(img_enc)

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(self.server_address)

                # Enviar el tamaño de la imagen comprimida y la imagen
                s.sendall(len(img_compressed).to_bytes(4, byteorder='little'))
                s.sendall(img_compressed)

                # Recibir la imagen procesada del servidor
                img_size_data = s.recv(4)
                if not img_size_data:
                    self.get_logger().error("Error: No se recibió el tamaño de la imagen del servidor.")
                    return

                img_size = int.from_bytes(img_size_data, byteorder='little')
                b_json = bytearray()

                # Recibir la imagen completa
                while len(b_json) < img_size:
                    data = s.recv(4096)
                    if not data:
                        self.get_logger().error("Error: Conexión interrumpida mientras se recibían datos.")
                        return
                    b_json.extend(data)

        except Exception as e:
            self.get_logger().error(f"Error en la conexión con el servidor: {e}")
            return

        if not b_json:
            self.get_logger().error("Error: Se recibió un búfer vacío del servidor.")
            return

        img_results = np.frombuffer(b_json, dtype=np.uint8)
        img_decoded = cv.imdecode(img_results, cv.IMREAD_COLOR)
        if img_decoded is None:
            self.get_logger().error("Error: No se pudo decodificar la imagen.")
            return

        # Mostrar la imagen procesada
        cv.namedWindow("Resultado", cv.WINDOW_NORMAL)  # Permitir redimensionamiento
        cv.resizeWindow("Resultado", 800, 600)  # Ajustar el tamaño de la ventana
        cv.imshow("Resultado", img_decoded)
        cv.waitKey(1)  # Esperar 1 ms

def main(args=None):
    rclpy.init(args=args)
    camara_client = CamaraClient()

    try:
        rclpy.spin(camara_client)
    except KeyboardInterrupt:
        pass
    finally:
        camara_client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


