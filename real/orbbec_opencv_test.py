import pygame
import qrcode
import time
import numpy as np
from collections import deque
import cv2


def generate_qr_code(qr_size=720):
    # Initialize QRCode detector (not used in this case, but can be used if decoding is needed)
    detector = cv2.QRCodeDetector()

    # Initialize Pygame
    pygame.init()

    # Create a window to display the QR Code
    window = pygame.display.set_mode((qr_size, qr_size))
    pygame.display.set_caption("Timestamp QRCode")

    # Prepare deque to store latencies
    qr_latency_deque = deque(maxlen=60)

    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get current timestamp to encode in QR code
        t_sample = time.time()

        # Create QRCode with timestamp data
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
        )
        qr.add_data(str(t_sample))
        qr.make(fit=True)
        pil_img = qr.make_image()

        # Convert the PIL image to numpy array and convert it to RGB format
        img = np.array(pil_img).astype(np.uint8) * 255
        img = np.repeat(img[:, :, None], 3, axis=-1)  # Convert to RGB (3 channels)
        img = cv2.resize(img, (qr_size, qr_size), cv2.INTER_NEAREST)

        # Convert the numpy array (image) to a Pygame surface
        surface = pygame.surfarray.make_surface(img)

        # Display the QR Code in the Pygame window
        window.blit(surface, (0, 0))
        pygame.display.update()

        # Log latency for QR generation
        t_show = time.time()
        qr_latency_deque.append(t_show - t_sample)

        # Poll for key presses (e.g., to quit the loop)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:  # Press 'q' to quit
            running = False

    # After loop ends, calculate and print latency stats
    avg_latency = np.mean(qr_latency_deque)
    print(f"Average QR Code generation latency: {avg_latency:.4f} seconds")

    pygame.quit()


if __name__ == "__main__":
    generate_qr_code()
