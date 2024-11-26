import torch
import torch.nn.functional as F


class AdversarialReg(object):
    def __init__(
        self,
        model,
        epsilon: float = 1e-5,
        lambda_: float = 5,
        eta: float = 1e-3,
        sigma: float = 1e-5,
        K: int = 1,
    ):
        # gọi hàm khởi tạo của lớp cha
        super(AdversarialReg, self).__init__()
        self.embed_backup = (
            {}
        )  # từ điển lưu trữ các tham số embedding ban đầu (sau cần khôi phục)
        self.grad_backup = (
            {}
        )  # từ điển lưu trữ các gradient ban đầu (sau khôi phục được)
        self.model = model
        self.epsilon = epsilon  # mức độ thay đổi các tham số
        self.lambda_ = lambda_  # hệ số điều chỉnh tính toán hàm mất mát
        self.eta = eta  # learning rate
        self.sigma = sigma  # độ mạnh của noise được thêm vào
        self.K = K  # iterations

    # Lưu các gradient được dùng tính toán (để sau khôi phục lại)
    def save_gradients(self):
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
            ):  # kiểm tra param có được tính toán trong backpropagation k
                if param.grad == None:  # chưa được tính toán
                    self.grad_backup[name] = None
                else:
                    self.grad_backup[name] = param.grad.clone()

    # lọc những tham số tgia tính toán là emb_name và lưu lại
    def save_embeddings(self, emb_name):
        # print(emb_name, type(emb_name))
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.embed_backup[name] = param.data.clone()

    # khôi phục các gradient tính toán
    def restore_gradient(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.grad_backup  # đảm bảo tham số tồn tại
                param.grad = self.grad_backup[name]  # khôi phục giá trị

    # khôi phục những tham số tính toán emb_name
    def restore_embeddings(self, emb_name):

        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.embed_backup
                param.data = self.embed_backup[name]
        self.embed_backup = {}

    # thêm nhiễu vào các tham số
    def generate_noise(self, emb_name):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                noise = param.data.new(param.size()).normal_(0, 1) * self.sigma
                param.data.add_(noise)

    # điều chỉnh tham số nhúng sau khi bị change mà đảm bảo không vượt quá ngưỡng xđ vởi epsilon
    def project(self, param_name, param_data):
        change = param_data - self.embed_backup[param_name]
        change = torch.clamp(change, min=-self.epsilon, max=self.epsilon)
        return self.embed_backup[param_name] + change

    # điều chỉnh tham số nhúng theo hướng gradient ascent (not GD)
    def emb_ascent(self, emb_name):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                norm = torch.norm(param.grad, p=float("inf"))
                if norm != 0 and not torch.isnan(norm):
                    param.data.add_(self.eta * (param.grad / norm))
                    param.data = self.project(name, param.data)

    # Tính toán KL divergence đối xứng -> sử dụng trong quá trình train
    def symmetric_kl(self, inputs, target):
        loss = F.kl_div(
            F.log_softmax(inputs, dim=-1),
            F.log_softmax(target, dim=-1),
            reduction="batchmean",
            log_target=True,
        )
        loss += F.kl_div(
            F.log_softmax(target, dim=-1),
            F.log_softmax(inputs, dim=-1),
            reduction="batchmean",
            log_target=True,
        )
        return loss

    # Một cách tính toán D_KL tương tự -> dùng cho việc đánh giá hiệu suất (val, test?)
    def symmetric_kl_check(self, inputs, target, reduce=True):
        epsilon = 1e-6
        bs = inputs.size(0)
        p = F.log_softmax(inputs, 1).exp()
        y = F.log_softmax(target, 1).exp()
        rp = -(1.0 / (p + epsilon) - 1 + epsilon).detach().log()
        ry = -(1.0 / (y + epsilon) - 1 + epsilon).detach().log()

        if reduce:
            return (p * (rp - ry) * 2).sum() / bs
        else:
            return (p * (rp - ry) * 2).sum()

    def max_loss_reg(self, batch, emb_name):  # embedding or embedding.
        emb_name = "embedding"
        # print(emb_name, type(emb_name))
        input_ids, attention_mask = batch
        self.model.eval()  # chuyển model sang chế độ đánh giá, tắt dropout

        logits = self.model(input_ids, attention_mask)  # tính toán f ban đầu
        # print(logits, type(logits))

        self.save_gradients()
        self.save_embeddings(emb_name)

        self.generate_noise(emb_name)

        for _ in range(self.K):
            self.model.zero_grad()

            adv_logits = self.model(
                input_ids, attention_mask
            )  # Tính toán f sau khi có thêm nhiễu
            # print(adv_logits, type(adv_logits))
            adv_loss = self.symmetric_kl(adv_logits, logits)

            adv_loss.backward()
            self.emb_ascent(emb_name)

        self.restore_gradient()

        adv_logits = self.model(
            input_ids, attention_mask
        )  # model_prediction(self.model, batch, task_name)
        adv_loss = self.symmetric_kl(adv_logits, logits.detach())

        self.restore_embeddings(emb_name)
        self.model.train()

        return self.lambda_ * adv_loss
