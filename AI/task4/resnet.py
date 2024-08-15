import numpy as np

EPS = 0.008


class MyReLU:
    def __init__(self):
        self.x = None
        self.x_grad = None

    def forward(self, x):
        self.x = x
        return np.maximum(0, self.x)

    def backward(self, g):
        if self.x_grad is None:
            self.x_grad = np.zeros_like(self.x)
        self.x_grad += (self.x > 0) * g

    def zero_grad(self):
        self.x_grad = None

    def __call__(self, x):
        return self.forward(x)


class MyConv2d:
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=0):
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding

        self.weights = np.random.rand(c_out, c_in, kernel_size, kernel_size) * EPS
        self.bias = np.random.rand(c_out) * EPS

        self.x = None
        self.x_grad = None
        self.w_grad = np.zeros_like(self.weights)
        self.b_grad = np.zeros_like(self.bias)

    def forward(self, x):
        self.x = x
        batch_size, c_in, h_in, w_in = x.shape

        h_out = int((h_in + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0] + 1)
        w_out = int((w_in + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1] + 1)

        x_padded = np.pad(x, ((0, 0), (0, 0), self.padding, self.padding), mode='constant')
        output = np.zeros((batch_size, self.c_out, h_out, w_out))

        for i in range(h_out):
            for j in range(w_out):
                x_slice = x_padded[:, :, i * self.stride[0]:i * self.stride[0] + self.kernel_size[0],
                                         j * self.stride[1]:j * self.stride[1] + self.kernel_size[1]]
                for k in range(self.c_out):
                    output[:, k, i, j] = np.sum(x_slice * self.weights[k], axis=(1, 2, 3)) + self.bias[k]

        return output

    def backward(self, g):
        if self.x_grad is None:
            self.x_grad = np.zeros_like(self.x)
        batch_size, c_out, h_out, w_out = g.shape

        x_padded = np.pad(self.x, ((0, 0), (0, 0), self.padding, self.padding), mode='constant')

        x_padded_grad = np.zeros_like(x_padded)
        for i in range(h_out):
            for j in range(w_out):
                x_padded_grad[:, :, i * self.stride[0]:i * self.stride[0] + self.kernel_size[0],
                                    j * self.stride[1]:j * self.stride[1] + self.kernel_size[1]] += np.tensordot(g[:, :, i, j], self.weights, axes=(1, 0))
                x_slice = x_padded[:, :, i * self.stride[0]:i * self.stride[0] + self.kernel_size[0],
                                         j * self.stride[1]:j * self.stride[1] + self.kernel_size[1]]
                self.w_grad += np.tensordot(g[:, :, i, j], x_slice, axes=(0, 0))
        if self.padding[0]:
            self.x_grad += x_padded_grad[:, :, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]]
        else:
            self.x_grad += x_padded_grad
        self.b_grad += np.ones_like(self.bias)

    def zero_grad(self):
        self.x_grad = None
        self.w_grad = np.zeros_like(self.weights)
        self.b_grad = np.zeros_like(self.bias)

    def __call__(self, x):
        return self.forward(x)


class MyMaxPool:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.x = None
        self.x_grad = None

    def forward(self, x):
        self.x = x
        batch_size, c_in, h_in, w_in = x.shape

        h_out = int((h_in - self.kernel_size[0]) / self.stride[0] + 1)
        w_out = int((w_in - self.kernel_size[1]) / self.stride[1] + 1)

        output = np.zeros((batch_size, c_in, h_out, w_out))

        for i in range(h_out):
            for j in range(w_out):
                x_slice = x[:, :, i * self.stride[0]:i * self.stride[0] + self.kernel_size[0],
                                  j * self.stride[1]:j * self.stride[1] + self.kernel_size[1]]
                output[:, :, i, j] = x_slice.max(axis=(2, 3))

        return output

    def backward(self, g):
        batch_size, c_in, h_out, w_out = g.shape
        if self.x_grad is None:
            self.x_grad = np.zeros_like(self.x)

        for i in range(h_out):
            for j in range(w_out):
                x_slice = self.x[:, :, i * self.stride[0]:i * self.stride[0] + self.kernel_size[0],
                                       j * self.stride[1]:j * self.stride[1] + self.kernel_size[1]]
                mask_slice = x_slice == x_slice.max(axis=(2, 3), keepdims=True)
                self.x_grad[:, :, i * self.stride[0]:i * self.stride[0] + self.kernel_size[0],
                                  j * self.stride[1]:j * self.stride[1] + self.kernel_size[1]] += mask_slice

    def zero_grad(self):
        self.x_grad = None

    def __call__(self, x):
        return self.forward(x)


class MyLayer:
    def __init__(self, c_in, c_out):
        self.conv1 = MyConv2d(c_in, c_in, 3, padding=1)
        self.relu1 = MyReLU()
        self.conv2 = MyConv2d(c_in, c_in, 3, padding=1)
        self.relu2 = MyReLU()
        self.conv3 = MyConv2d(c_in, c_out, 3, padding=1)
        self.relu3 = MyReLU()
        self.skip_conv = MyConv2d(c_in, c_out, 1)
        self.maxpool = MyMaxPool(2, 2)
        self.x = None
        self.x_grad = None

    def forward(self, x):
        self.x = x
        skip = self.x
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        skip = self.skip_conv(skip)
        x = x + skip
        x = self.maxpool(x)
        return x

    def backward(self, g):
        if self.x_grad is None:
            self.x_grad = np.zeros_like(self.x)
        for module in self.modules()[::-1]:
            if isinstance(module, list):
                for submodule in module:
                    submodule.backward(g)
                g = module[0].x_grad
                g_skip = module[1].x_grad
            else:
                module.backward(g)
                g = module.x_grad
        self.x_grad += g + g_skip

    def zero_grad(self):
        self.x_grad = None

    def modules(self):
        return [self.conv1, self.relu1, self.conv2, self.relu2, self.conv3, [self.relu3, self.skip_conv], self.maxpool]

    def __call__(self, x):
        return self.forward(x)


class MyFC:
    def __init__(self, c_in, c_out):
        self.weights = np.random.rand(c_out, c_in) * EPS
        self.bias = np.random.rand(c_out) * EPS
        self.x = None
        self.x_grad = None
        self.w_grad = np.zeros_like(self.weights)
        self.b_grad = np.zeros_like(self.bias)

    def forward(self, x):
        self.x = x
        output = x @ self.weights.T + self.bias
        return output

    def backward(self, g):
        if self.x_grad is None:
            self.x_grad = np.zeros_like(self.x)
        self.x_grad += g @ self.weights
        self.w_grad += g.T @ self.x
        self.b_grad += g.sum(axis=0)

    def zero_grad(self):
        self.x_grad = None
        self.w_grad = np.zeros_like(self.weights)
        self.b_grad = np.zeros_like(self.bias)

    def __call__(self, x):
        return self.forward(x)


class MySoftMax:
    def __init__(self):
        self.output = None
        self.x_grad = None

    def forward(self, x):
        exp_x = np.exp(x)
        sums = exp_x.sum(axis=1, keepdims=True)
        self.output = exp_x / sums
        return self.output

    def backward(self, g):
        if self.x_grad is None:
            self.x_grad = np.zeros_like(self.output)
        self.x_grad += self.output * (1 - self.output) * g

    def zero_grad(self):
        self.x_grad = None

    def __call__(self, x):
        return self.forward(x)


class MySqueeze:
    def __init__(self):
        self.x = None
        self.x_grad = None

    def forward(self, x):
        self.x = x
        return self.x.squeeze()

    def backward(self, g):
        if self.x_grad is None:
            self.x_grad = np.zeros_like(self.x)
        self.x_grad += g[:, :, None, None]

    def zero_grad(self):
        self.x_grad = None

    def __call__(self, x):
        return self.forward(x)


class MyResNet:
    def __init__(self):
        self.layer1 = MyLayer(3, 16)
        self.layer2 = MyLayer(16, 32)
        self.layer3 = MyLayer(32, 64)
        self.layer4 = MyLayer(64, 128)
        self.layer5 = MyLayer(128, 256)
        self.squeeze = MySqueeze()
        self.fc = MyFC(256, 10)
        self.softmax = MySoftMax()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.squeeze(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

    def backward(self, g):
        for module in self.modules()[::-1]:
            module.backward(g)
            g = module.x_grad

    def modules(self):
        return [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.squeeze, self.fc, self.softmax]

    def __call__(self, x):
        return self.forward(x)


class MyCrossEntropyLoss:
    def __init__(self):
        self.gt = None
        self.pred = None
        self.pred_grad = None

    def forward(self, gt, pred):
        self.gt = gt
        self.pred = pred
        loss = -(gt * np.log(pred)).mean()
        return loss

    def backward(self, g):
        if self.pred_grad is None:
            self.pred_grad = np.zeros_like(self.pred)
        self.pred_grad += self.gt / self.pred * g

    def __call__(self, gt, pred):
        return self.forward(gt, pred)


def train_loop(model, batches, loss_fn, optimizer):
    for i, (images, labels) in enumerate(batches):
        pred = model(images)
        loss = loss_fn(labels, pred)
        model.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        accuracy = (np.where(pred > 0.5, 1, 0) == labels).sum() / (labels.shape[0] * labels.shape[1])
        if i % 100 == 0:
            print(f'loss = {loss:.3f}, accuracy = {accuracy:.3f}')


def val_loop(model, batches, loss_fn):
    for i, (images, labels) in enumerate(batches):
        pred = model(images)
        loss = loss_fn(labels, pred)
        accuracy = (np.where(pred > 0.5, 1, 0) == labels).sum() / (labels.shape[0] * labels.shape[1])
        if i % 100 == 0:
            print(f'loss = {loss:.3f}, accuracy = {accuracy:.3f}')


def train(model, epochs, batches, optimizer, loss_fn):
    for epoch in range(epochs):
        train_loop(model, batches, loss_fn, optimizer)
        # val_loop(model, batches, loss_fn)


class MySGD:
    def __init__(self, model, lr=1e-3, weight_decay=5e-3):
        self.modules = model.modules()
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        for module in self.modules:
            if getattr(module, 'modules', None) is None:
                if getattr(module, 'w_grad', None) is not None:
                    module.weights -= self.lr * (module.w_grad + self.weight_decay * module.weights)
                if getattr(module, 'b_grad', None) is not None:
                    module.bias -= self.lr * (module.b_grad + self.weight_decay * module.bias)
            else:
                for submodule in module.modules():
                    if getattr(submodule, 'w_grad', None) is not None:
                        submodule.weights -= self.lr * (submodule.w_grad + self.weight_decay * submodule.weights)
                    if getattr(submodule, 'b_grad', None) is not None:
                        submodule.bias -= self.lr * (submodule.b_grad + self.weight_decay * submodule.bias)

    def zero_grad(self):
        for module in self.modules:
            if getattr(module, 'modules', None) is None:
                if getattr(module, 'zero_grad', None) is not None:
                    module.zero_grad()
            else:
                for submodule in module.modules():
                    if getattr(submodule, 'zero_grad', None) is not None:
                        submodule.zero_grad()


def main():
    batches = [(np.random.rand(5, 3, 32, 32), np.where(np.random.randn(5, 10) > 0.5, 1, 0))]
    resnet = MyResNet()
    optimizer = MySGD(resnet, lr=0.009)
    loss_fn = MyCrossEntropyLoss()
    train(resnet, 100, batches, optimizer, loss_fn)


if __name__ == '__main__':
    main()
