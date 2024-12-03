- Tệp ```src/adversarial.py``` chứa lớp ```AdversarialReg``` được sử dụng để áp dụng kỹ thuật regularization. Lớp này gồm các phương thức để lưu trữ và khôi phục các tham số nhúng và gradient, tạo nhiễu, điều chỉnh các tham số nhúng bằng gradient ascent, và tính toán hàm mất mát KL divergence đối xứng.
- Tệp ```src/dataset.py``` chứa các lớp xử lý các dataset khác nhau:  
  - ```SST5_Dataset```: Xử lý dữ liệu từ SST.  
  - ```Amazon_Dataset```: Xử lý dữ liệu từ Amazon.
  - ```NLI_Dataset```: Dùng cho học tương phản có giám sát với định dạng NLI.  
  - ```WikiDataset```: Dùng cho học tương phản không giám sát với dữ liệu từ wiki1m.  
- Tệp ```src/loss.py``` chứa hàm ```supCL_loss```, một hàm tính toán contrastive loss có giám sát. Hàm này sử dụng độ tương đồng cosine giữa các điểm dữ liệu anchor, positive và negative
- Tệp ```src/mbpp.py``` chứa lớp ```MBPP``` (Momentum Bregman Proximal Point), nhằm tránh việc cập nhật quá mức các tham số của mô hình. Lớp này bao gồm các phương pháp để cập nhật tham số mô hình theo quy tắc momentum, tính toán độ lệch Bregman giữa các phân phối xác suất, và khôi phục tham số mô hình về trạng thái ban đầu.
- Tệp ```src/model.py``` chứa lớp ```BertClassifier```, phân loại sử dụng BERT. Lớp này bao gồm một mô hình BERT, một lớp dropout, và một bộ phân loại bao gồm các lớp fully connected, ReLU, batch normalization, và dropout.
- Tệp ```src/test.py``` chứa hàm ```test_model``` để tải và kiểm tra mô hình trên tập dữ liệu thử nghiệm, tính toán loss, accuracy, precision, recall, F1 score và vẽ confusion matrix.
- Tệp ```src/train_supcl.py``` chứa hàm ```train_cl``` để thực hiện quá trình huấn luyện tương phản có giám sát, sử dụng các cặp dữ liệu và tính toán loss, sau đó lưu lại mô hình đã huấn luyện. Nó cũng bao gồm mã để tải dữ liệu huấn luyện từ các tập dữ liệu khác nhau, tạo các cặp dữ liệu.
- Tệp ```src/tran_unsupcl``` chứa hàm ```train_uncl``` để huấn luyện mô hình tương phản có giám sát. Tệp này cũng bao gồm mã để tải cấu hình, tạo dữ liệu từ wiki.
- Tệp ```src/utils.py``` chứa các hàm tiện ích để xử lý dữ liệu và cấu hình cho mô hình:
  - ```read_ptb_tree```, ```extract_sentence_and_label``` đọc và xử lý Treebank.
  - ```read_file``` đọc dữ liệu từ tệp SST.
  - ```group_data_by_level``` nhóm dữ liệu theo mức độ cảm xúc cho học tương phản có giám sát.
  - ```create_large_data_pairs``` tạo cặp dữ liệu cho học tương phản có giám sát.
  - ```canonicalize_text``` chuẩn hóa văn bản.
  - ```create_triples``` tạo bộ ba dữ liệu cho học tương phản có giám sát.
  - ```read_amazon_reviews``` đọc đánh giá từ Amazon.
  - ```load_config``` tải tệp cấu hình.








