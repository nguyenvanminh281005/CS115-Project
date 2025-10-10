# Image Generation with Denoising Diffusion Probabilistic Models (DDPM)

<div align="center">
  <img src="https://raw.githubusercontent.com/nguyenvanminh281005/cs115-project/main/Report_CS115_KHTN2023_DDPM.pdf-page1-body-image-0.jpg" alt="DDPM Process" width="700"/>
</div>

<p align="center">
  <em>Đây là một dự án nghiên cứu và triển khai mô hình <strong>Denoising Diffusion Probabilistic Models (DDPM)</strong> để sinh ảnh, là một phần của môn học CS115 - Toán cho Khoa học Máy tính.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/tensorflow-%23FF6F00.svg?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/jax-%235E48B3.svg?style=for-the-badge&logo=jax&logoColor=white" alt="JAX">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
</p>

## 🌟 Giới thiệu về DDPM

**Denoising Diffusion Probabilistic Models (DDPM)** là một lớp các mô hình sinh (Generative Models) trong học máy, được thiết kế để tạo ra dữ liệu mới bằng cách mô phỏng hai quá trình chính:

1.  **Quá trình khuếch tán thuận (Forward Process):** Một quá trình thêm nhiễu (noise) vào dữ liệu gốc một cách từ từ qua nhiều bước, cho đến khi dữ liệu trở thành nhiễu hoàn toàn theo phân phối Gaussian.
2.  **Quá trình đảo ngược (Reverse Process):** Mô hình học cách đảo ngược quá trình trên, tức là khử nhiễu (denoise) từ một mẫu nhiễu ngẫu nhiên để tái tạo lại một mẫu dữ liệu sạch, có cấu trúc.

Ý tưởng cốt lõi là biến một phân phối dữ liệu phức tạp thành một phân phối nhiễu đơn giản mà ta có thể dễ dàng lấy mẫu, sau đó học cách biến đổi ngược lại để sinh dữ liệu mới.

## ✨ Kết quả Thực nghiệm

Dự án đã triển khai thành công mô hình DDPM trên bộ dữ liệu **MNIST**. Dưới đây là hình ảnh minh họa quá trình khử nhiễu để tạo ra các chữ số viết tay từ nhiễu Gaussian hoàn toàn.

#### Quá trình sinh ảnh các chữ số

* **Bước 0:** Bắt đầu từ một ảnh nhiễu hoàn toàn.
* **Các bước trung gian:** Mô hình dần dần loại bỏ nhiễu, cấu trúc của chữ số bắt đầu hiện ra.
* **Bước cuối:** Ảnh nhiễu được chuyển hóa thành một ảnh chữ số rõ nét.

<p align="center">
  <em>(Hình ảnh minh họa quá trình sinh các chữ số '2', '7', và '3' từ nhiễu)</em>
  <img src="https://raw.githubusercontent.com/nguyenvanminh281005/cs115-project/main/Report_CS115_KHTN2023_DDPM.pdf-page12-body-image-0.jpg" alt="Generated Digits" width="800"/>
  <br>
  <em>Kết quả sau 10 epochs huấn luyện.</em>
</p>

[cite_start]Mô hình cũng được so sánh với các kiến trúc sinh ảnh khác như **GANs** và **VAEs** để đánh giá hiệu suất[cite: 20, 871].

## 🚀 Công nghệ và Kiến trúc

Mô hình được xây dựng chủ yếu bằng **Python** với sự hỗ trợ của các thư viện học sâu mạnh mẽ.

* **Frameworks:** **TensorFlow** (để xử lý dữ liệu) và **JAX/Flax** (để xây dựng và huấn luyện mô hình U-Net).
* **Kiến trúc Mạng:**
    * **U-Net:** Được sử dụng làm mạng neural chính để dự đoán nhiễu ở mỗi bước trong quá trình đảo ngược.
    * [cite_start]**Attention Mechanism:** Tích hợp các khối Attention để mô hình có thể tập trung vào các vùng quan trọng của ảnh[cite: 9].
    * [cite_start]**Time Embedding:** Sử dụng kỹ thuật Sinusoidal Embedding để mã hóa thông tin về bước thời gian (timestep) và đưa vào mô hình[cite: 8].

## 🔧 Cài đặt và Chạy mã nguồn

Bạn có thể chạy lại thực nghiệm này bằng cách làm theo các bước dưới đây.

### **1. Yêu cầu**

* Python 3.9+
* TensorFlow
* JAX và Flax
* Optax
* TensorFlow Datasets
* Matplotlib

### **2. Cài đặt môi trường**

Clone repository này về máy:
```bash
git clone [https://github.com/nguyenvanminh281005/cs115-project.git](https://github.com/nguyenvanminh281005/cs115-project.git)
cd CS115-Project
```
Cài đặt các thư viện cần thiết:

``` Bash

pip install tensorflow tensorflow-datasets jax flax optax matplotlib
```
3. Chạy thực nghiệm
Mã nguồn chính cho việc triển khai và huấn luyện mô hình nằm trong tệp Jupyter Notebook vming.ipynb.

Mở tệp vming.ipynb bằng Jupyter Notebook hoặc Jupyter Lab.

Chạy các cell theo thứ tự để:

Tải và tiền xử lý bộ dữ liệu MNIST.

Định nghĩa các thành phần của mô hình (U-Net, Attention, Time Embedding).

Thiết lập quá trình huấn luyện.

Chạy vòng lặp huấn luyện.

Sử dụng mô hình đã huấn luyện để sinh ảnh mới từ nhiễu và lưu kết quả dưới dạng GIF.

📚 Tài liệu tham khảo
Để hiểu sâu hơn về lý thuyết toán học đằng sau DDPM, vui lòng tham khảo các tài liệu sau trong repository:

Report_CS115_KHTN2023_DDPM.pdf: Báo cáo chi tiết về cơ sở lý thuyết, công thức toán học, và phân tích mô hình.

Slides_CS115_KHTN2023_DDPM.pdf: Bài trình bày tóm tắt các nội dung chính của dự án.

📄 Giấy phép
Dự án này được cấp phép theo Giấy phép MIT. Xem chi tiết tại tệp LICENSE.