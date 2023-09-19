from PIL.ImageFont import ImageFont
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PIL import Image, ImageDraw
import cv2
import os
import sys
import numpy as np


class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(512, 512)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setStyleSheet("background-color: lightgray; border: 2px dashed gray;")

        self.image1_rgb = None
        self.image1_mask = None
        self.image2_rgb = None
        self.image2_mask = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        event.accept()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = str(url.toLocalFile())
            if ("RGBMask" in file_path or "Normal" in file_path) and (
                file_path.lower().endswith(".jpg") or file_path.lower().endswith(".png")
            ):
                image = cv2.imread(file_path)
                if image is not None:
                    # Separate color channels using PIL
                    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    rgb_channels = pil_image.split()

                    if "RGBMask" in file_path:
                        if self.image1_rgb is None:
                            self.image1_rgb = rgb_channels[0]
                            self.image1_mask = rgb_channels[1]
                            self.setText("Image 1\nDropped")
                        elif self.image2_rgb is None:
                            self.image2_rgb = rgb_channels[0]
                            self.image2_mask = rgb_channels[1]
                            self.setText("Image 2\nDropped")
                        else:
                            break
                    elif "Normal" in file_path:
                        if self.image1_rgb is None:
                            self.image1_rgb = rgb_channels[0]
                            self.setText("Image 1\nDropped")
                        elif self.image2_rgb is None:
                            self.image2_rgb = rgb_channels[0]
                            self.setText("Image 2\nDropped")
                        else:
                            break

                    # Update the label preview
                    height, width = pil_image.size
                    bytes_per_line = 3 * width
                    q_image = QImage(
                        image.data, width, height, bytes_per_line, QImage.Format_RGB888
                    )
                    pixmap = QPixmap.fromImage(q_image.rgbSwapped())
                    self.setPixmap(
                        pixmap.scaled(
                            512, 512, Qt.KeepAspectRatio, Qt.SmoothTransformation
                        )
                    )

                    event.accept()
                    break
        else:
            event.ignore()


class DraggableLineEdit(QLineEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        url = event.mimeData().urls()[0].toLocalFile()
        self.setText(url)


class GridGenerator:
    def __init__(self, width, height, num_rows, num_cols, cell_size, alpha):
        self.width = width
        self.height = height
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.cell_size = cell_size
        self.alpha = alpha

        self.grid = self.generate_grid()

    def generate_grid(self):
        grid_image = Image.new(
            "RGBA",
            (self.width, self.height),
            color=(140, 140, 140, int(self.alpha * 255)),
        )
        draw = ImageDraw.Draw(grid_image)

        square_dict = {
            "Metallic": (0, 0),
            "Smoothness": (1, 0),
            "Emission": (2, 0),
            "Fresnel": (3, 0),
            "Fuzziness": (1, 1),
            "OGMtSmOn": (1, 1),
            "OGMtSmOff": (1, 1),
            "Red_Mask": (1, 2),
            "Blue_Mask": (1, 3),
            "Iridescent_Red": (2, 0),
            "Iridescent_Green": (2, 1),
            "Iridescent_Blue": (2, 2),
            "Iridescent_Mask": (2, 3),
            "Glitter_Red": (3, 0),
            "Glitter_Green": (3, 1),
            "Glitter_Blue": (3, 2),
            "Glitter_Mask": (3, 3),
        }

        for name, (row, col) in square_dict.items():
            x = col * self.cell_size
            y = row * self.cell_size

            # Draw the square
            draw.rectangle(
                [x, y, x + self.cell_size, y + self.cell_size],
                fill=(140, 140, 140, int(self.alpha * 255)),
            )

            # Add label text
            label_text = f"{name} ({row}, {col}) {int(self.alpha * 100)}%"
            draw.text(
                (x + 2, y + 2),
                label_text,
                fill=(0, 255, 255, 255),
                font=ImageFont.truetype("arial.ttf", 10),
            )

        return grid_image


class TextureGenerator(QMainWindow):
    def __init__(self):
        super().__init__()

        self.current_square = None

        # Set up UI
        self.setWindowTitle("Shader Texture Mapper")
        layout = QVBoxLayout()

        self.setCentralWidget(ImageLabel())

        self.preview_label = QLabel()
        self.preview_label.setFixedSize(512, 512)
        layout.addWidget(self.preview_label)

        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(
            0, 100
        )  # Scale the slider range to 0-100 for alpha values between 0 and 1
        self.alpha_slider.valueChanged.connect(self.update_alpha)
        self.alpha_slider.setValue(50)

        layout.addWidget(self.alpha_slider)

        combo_layout = QHBoxLayout()
        self.square_selector = QComboBox()
        self.square_selector.addItems(
            ["Metallic", "Smoothness", "Emission", "Fresnel", "Fuzziness"]
        )
        self.square_selector.currentTextChanged.connect(self.update_current_square)
        self.square_selector.currentIndexChanged.connect(
            self.on_square_selection_changed
        )
        self.square_selector.currentIndexChanged.connect(self.update_current_square)
        # self.layout.addWidget(self.square_selector, 1, 0, 1, 2)
        combo_layout.addWidget(self.square_selector)

        self.combo_box2 = QComboBox()
        self.combo_box2.addItems(["On", "Off"])
        self.combo_box2.currentTextChanged.connect(self.toggle_og_mtsm)
        combo_layout.addWidget(self.combo_box2)

        layout.addLayout(combo_layout)

        picker_layout = QHBoxLayout()

        # Push Buttons for Iridescent and Glitter
        self.iridescent_button = QPushButton("Iridescent")
        self.iridescent_button.clicked.connect(lambda: self.color_picker("Iridescent"))
        picker_layout.addWidget(self.iridescent_button)

        self.glitter_button = QPushButton("Glitter")
        self.glitter_button.clicked.connect(lambda: self.color_picker("Glitter"))
        picker_layout.addWidget(self.glitter_button)

        layout.addLayout(picker_layout)

        self.path_input = DraggableLineEdit("Drag output folder here")
        layout.addWidget(self.path_input)

        self.create_button = QPushButton("Create Texture")
        self.create_button.clicked.connect(self.create_texture)
        layout.addWidget(self.create_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Generate initial texture
        self.current_square = "Metallic"
        self.generate_texture()

    def on_square_selection_changed(self, index):
        if index == -1:
            return

        selected_square = self.square_selector.itemText(index)
        _, _, _, alpha = self.square_dict[selected_square]
        self.alpha_slider.blockSignals(True)
        self.alpha_slider.setValue(int(alpha * 100))
        self.alpha_slider.blockSignals(False)

    def update_current_square(self, index):
        self.current_square = self.square_selector.currentText()

    def generate_texture(self):
        grid_size = 4
        square_size = 128

        square_names = [
            "Metallic",
            "Smoothness",
            "Emission",
            "Fresnel",
            "Fuzziness",
            "OGMtSm",
            "Red_Mask",
            "Blue_Mask",
            "Iridescent_Red",
            "Iridescent_Green",
            "Iridescent_Blue",
            "Iridescent_Mask",
            "Glitter_Red",
            "Glitter_Green",
            "Glitter_Blue",
            "Glitter_Mask",
        ]

        self.square_dict = {name: (140, 140, 140, 0.5) for name in square_names}
        self.square_dict["OGMtSm"] = (140, 140, 140, 0)
        self.square_dict["Iridescent_Mask"] = (140, 140, 140, 1)
        self.square_dict["Glitter_Mask"] = (140, 140, 140, 1)

        image = np.zeros(
            (grid_size * square_size, grid_size * square_size, 4), dtype=np.uint8
        )

        for i, name in enumerate(square_names):
            x = i % grid_size
            y = i // grid_size
            gray, _, _, alpha = self.square_dict[name]
            color = (gray, gray, gray, int(alpha * 255))
            image[
                y * square_size : (y + 1) * square_size,
                x * square_size : (x + 1) * square_size,
            ] = color

        self.preview_image = image
        self.update_preview()

    def update_preview(self):
        grid_size = 4
        square_size = 128

        image = np.zeros(
            (grid_size * square_size, grid_size * square_size, 4), dtype=np.uint8
        )

        for i, name in enumerate(self.square_dict.keys()):
            x = i % grid_size
            y = i // grid_size
            gray, _, _, alpha = self.square_dict[name]
            color = (gray, gray, gray, int(alpha * 255))
            image[
                y * square_size : (y + 1) * square_size,
                x * square_size : (x + 1) * square_size,
            ] = color

            # Add text labels
            text = f"{name}" + f"{alpha:.2f}"
            cv2.putText(
                img=image,
                text=text,
                org=(x * square_size + 5, y * square_size + 20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.3,
                color=(128, 128, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        self.preview_image = image

        # Convert the image to QPixmap and update the preview label
        qimage = QImage(
            image.data, image.shape[1], image.shape[0], QImage.Format_RGBA8888
        )
        pixmap = QPixmap.fromImage(qimage)

        self.preview_label.setPixmap(pixmap)

    def update_current_square(self, square_name):
        self.current_square = square_name

    def update_alpha(self, value):
        alpha = value / 100

        if self.current_square:
            gray, _, _, _ = self.square_dict[self.current_square]
            self.square_dict[self.current_square] = (gray, gray, gray, alpha)

            self.update_preview()

    def toggle_og_mtsm(self, index):
        og_mtsm = "OGMtSm"
        gray, _, _, _ = self.square_dict[og_mtsm]

        if index == 0:  # OGMtSmOn
            self.square_dict[og_mtsm] = (gray, gray, gray, 1)
        elif index == 1:  # OGMtSmOff
            self.square_dict[og_mtsm] = (gray, gray, gray, 0)

        self.update_preview()

    def create_texture(self):
        if not self.path_input.text():
            QMessageBox.warning(
                self,
                "No folder selected",
                "Please select a folder before generating a texture.",
            )
            return

        output_image = self.preview_image.copy()
        output_image = cv2.resize(
            output_image, (2048, 2048), interpolation=cv2.INTER_LINEAR
        )

        # Comment out the line that draws the names and alpha values
        # cv2.putText(output_image, name + f' {alpha}', (x, y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0, 255), 1, cv2.LINE_AA)

        output_path = os.path.join(self.path_input.text(), "output_texture_001.png")
        cv2.imwrite(output_path, output_image)
        QMessageBox.information(
            self,
            "Texture generated",
            f"Texture successfully generated at {output_path}",
        )

    def color_picker(self, button):
        color = QColorDialog.getColor()

        if color.isValid():
            r, g, b, _ = color.getRgb()
            gray_r = int(0.299 * r + 0.587 * g + 0.114 * b)

            if button == "Iridescent":
                prefixes = ["Iridescent_Red", "Iridescent_Green", "Iridescent_Blue"]
            else:
                prefixes = ["Glitter_Red", "Glitter_Green", "Glitter_Blue"]

            for prefix, channel in zip(prefixes, [r, g, b]):
                gray, _, _, _ = self.square_dict[prefix]
                self.square_dict[prefix] = (gray, gray, gray, channel / 255)
            self.update_preview()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = TextureGenerator()
    window.show()

    sys.exit(app.exec())
