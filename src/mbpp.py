import torch
import torch.nn.functional as F

# Source code from : https://github.com/jelc53/nlp-minibert

# Định nghĩa lớp momentum Bregman proximal point để tránh tham số bị cập nhật quá mức
class MBPP(object):
    def __init__(self, model, beta: float = 0.995, mu: float = 1):
        self.model = model
        self.beta = beta
        self.mu = mu
        self.theta_state = {}  # lưu trữ giá trị của tham số tại từng bước cập nhật

        # Cập nhật theta_0 bằng tham số model sau khi pre-trained
        for name, param in self.model.named_parameters():
            self.theta_state[name] = param.data.clone()

    # Cập nhật tất cả tham số theo quy tắc
    def apply_momentum(self, named_parameters):
        for name, param in self.model.named_parameters():
            self.theta_state[name] = (
                1 - self.beta
            ) * param.data.clone() + self.beta * self.theta_state[name]

    # Tính D_Breg
    def bregman_divergence(self, batch, logits):
        input_ids, attention_mask = batch
        # chuyển logits thành xác suất cho từng lớp bằng softmax
        theta_prob = F.softmax(logits, dim=-1)

        # sao lưu tham số hiện tại của mô hình (param.data) vào param_bak
        # thay thế bằng tham số lưu ở theta_state để tính toán mà k mất tham số gốc
        param_bak = {}
        for name, param in self.model.named_parameters():
            param_bak[name] = param.data.clone()
            param.data = self.theta_state[name]

        with torch.no_grad():
            logits = self.model(
                input_ids, attention_mask
            )  # model_prediction(self.model, batch, taskname)
            theta_til_prob = F.softmax(logits, dim=-1).detach()

        # khôi phục lại tham số
        for name, param in self.model.named_parameters():
            param.data = param_bak[name]

        # torch.cuda.empty_cache()

        # Tính toán
        l_s = F.kl_div(
            theta_prob.log(), theta_til_prob, reduction="batchmean"
        ) + F.kl_div(theta_til_prob.log(), theta_prob, reduction="batchmean")

        return self.mu * l_s
