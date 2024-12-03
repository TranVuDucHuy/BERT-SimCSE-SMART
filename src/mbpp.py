import torch
import torch.nn.functional as F

# Source code from : https://github.com/jelc53/nlp-minibert

# Định nghĩa lớp momentum Bregman proximal point để tránh tham số bị cập nhật quá mức
class MBPP(object):
    def __init__(self, model, beta: float = 0.995, mu: float = 1):
        self.model = model
        self.beta = beta
        self.mu = mu
        self.theta_state = {}                      # lưu trữ giá trị của tham số tại từng bước cập nhật

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
        theta_prob = F.softmax(logits, dim=-1)      # chuyển logits thành xác suất cho từng lớp bằng softmax
        
        param_bak = {}
        for name, param in self.model.named_parameters():
            param_bak[name] = param.data.clone()    # sao lưu tham số hiện tại của mô hình (param.data) vào param_bak
            param.data = self.theta_state[name]     # thay thế bằng tham số lưu ở theta_state để tính toán mà không mất tham số gốc

        with torch.no_grad():
            logits = self.model(
                input_ids, attention_mask
            ) 
            theta_til_prob = F.softmax(logits, dim=-1).detach()

        # Khôi phục lại tham số
        for name, param in self.model.named_parameters():
            param.data = param_bak[name]

        # Tính toán loss
        l_s = F.kl_div(
            theta_prob.log(), theta_til_prob, reduction="batchmean"
        ) + F.kl_div(theta_til_prob.log(), theta_prob, reduction="batchmean")

        return self.mu * l_s
