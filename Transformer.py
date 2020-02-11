# coding=utf-8
# パッケージのimport
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from utils.dataloader import get_IMDb_DataLoaders_and_TEXT
from utils.transformer import TransformerClassification
import torchtext
from IPython.display import HTML

# 乱数のシードを設定
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


class Transformer():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ネットワークの初期化を定義
    @staticmethod
    def __weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            # Liner層の初期化
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    @staticmethod
    def train_model(net, data_loaders_dict, criterion, optimizer, num_epochs):

        # GPUが使えるかを確認
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("使用デバイス：", device)
        print('-----start-------')
        # ネットワークをGPUへ
        net.to(device)

        # ネットワークがある程度固定であれば、高速化させる
        torch.backends.cudnn.benchmark = True

        # epochのループ
        for epoch in range(num_epochs):
            print("Start epoch {}".format(epoch))
            # epochごとの訓練と検証のループ
            for phase in ['train', 'val']:
                if phase == 'train':
                    net.train()  # モデルを訓練モードに
                else:
                    net.eval()  # モデルを検証モードに

                epoch_loss = 0.0  # epochの損失和
                epoch_corrects = 0  # epochの正解数

                # データローダーからミニバッチを取り出すループ
                for batch in (data_loaders_dict[phase]):
                    # batchはTextとLableの辞書オブジェクト

                    # GPUが使えるならGPUにデータを送る
                    inputs = batch.Text[0].to(device)  # 文章
                    labels = batch.Label.to(device)  # ラベル

                    # optimizerを初期化
                    optimizer.zero_grad()

                    # 順伝搬（forward）計算
                    with torch.set_grad_enabled(phase == 'train'):

                        # mask作成
                        input_pad = 1  # 単語のIDにおいて、'<pad>': 1 なので
                        input_mask = (inputs != input_pad)

                        # Transformerに入力
                        outputs, _, _ = net(inputs, input_mask)
                        loss = criterion(outputs, labels)  # 損失を計算

                        _, preds = torch.max(outputs, 1)  # ラベルを予測

                        # 訓練時はバックプロパゲーション
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                        # 結果の計算
                        epoch_loss += loss.item() * inputs.size(0)  # lossの合計を更新
                        # 正解数の合計を更新
                        epoch_corrects += torch.sum(preds == labels.data)

                # epochごとのlossと正解率
                epoch_loss = epoch_loss / len(data_loaders_dict[phase].dataset)
                epoch_acc = epoch_corrects.double(
                ) / len(data_loaders_dict[phase].dataset)

                print('Epoch {}/{} | {:^5} |  Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, num_epochs,
                                                                               phase, epoch_loss, epoch_acc))

        return net

    def evaluate_by_test_dataset(self, net_trained, test_dl):
        # device
        device = self.device

        net_trained.eval()  # モデルを検証モードに
        net_trained.to(device)

        epoch_corrects = 0  # epochの正解数

        for batch in test_dl:  # testデータのDataLoader
            # batchはTextとLableの辞書オブジェクト

            # GPUが使えるならGPUにデータを送る
            inputs = batch.Text[0].to(device)  # 文章
            labels = batch.Label.to(device)  # ラベル

            # 順伝搬（forward）計算
            with torch.set_grad_enabled(False):
                # mask作成
                input_pad = 1  # 単語のIDにおいて、'<pad>': 1 なので
                input_mask = (inputs != input_pad)

                # Transformerに入力
                outputs, _, _ = net_trained(inputs, input_mask)
                _, preds = torch.max(outputs, 1)  # ラベルを予測

                # 結果の計算
                # 正解数の合計を更新
                epoch_corrects += torch.sum(preds == labels.data)

        # 正解率
        epoch_acc = epoch_corrects.double() / len(test_dl.dataset)

        print('テストデータ{}個での正解率：{:.4f}'.format(len(test_dl.dataset), epoch_acc))

    def execute(self):
        # 読み込み
        train_dl, val_dl, test_dl, TEXT = get_IMDb_DataLoaders_and_TEXT(
            max_length=256, batch_size=64)

        # 辞書オブジェクトにまとめる
        dataloaders_dict = {"train": train_dl, "val": val_dl}

        # モデル構築
        net = TransformerClassification(
            text_embedding_vectors=TEXT.vocab.vectors, d_model=300, max_seq_len=256, output_dim=2)

        net.train()

        # TransformerBlockモジュールを初期化実行
        net.net3_1.apply(self.__weights_init)
        net.net3_2.apply(self.__weights_init)

        print('ネットワーク設定完了')
        # 損失関数の設定

        criterion = nn.CrossEntropyLoss()
        # nn.LogSoftmax()を計算してからnn.NLLLoss(negative log likelihood loss)を計算

        # 最適化手法の設定
        learning_rate = 2e-5
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)

        # 学習・検証を実行する 15分ほどかかります
        num_epochs = 10
        net_trained = self.train_model(net=net, data_loaders_dict=dataloaders_dict,
                                       criterion=criterion, optimizer=optimizer, num_epochs=num_epochs)

        torch.save(net_trained, "net_trained.weights")

        # テストデータでの正解率を求める
        self.evaluate_by_test_dataset(net_trained=net_trained, test_dl=test_dl)

        # Transformerで処理

        # ミニバッチの用意
        batch = next(iter(test_dl))

        # GPUが使えるならGPUにデータを送る
        inputs = batch.Text[0].to(self.device)  # 文章
        labels = batch.Label.to(self.device)  # ラベル

        # mask作成
        input_pad = 1  # 単語のIDにおいて、'<pad>': 1 なので
        input_mask = (inputs != input_pad)

        # Transformerに入力
        outputs, normalized_weights_1, normalized_weights_2 = net_trained(
            inputs, input_mask)
        _, preds = torch.max(outputs, 1)  # ラベルを予測

        index = 3  # 出力させたいデータ
        html_output = self.mk_html(index, batch, preds, normalized_weights_1,
                                   normalized_weights_2, TEXT)  # HTML作成
        HTML(html_output)  # HTML形式で出力

    # HTMLを作成する関数を実装
    @staticmethod
    def __highlight(word, attn):
        "Attentionの値が大きいと文字の背景が濃い赤になるhtmlを出力させる関数"

        html_color = '#%02X%02X%02X' % (
            255, int(255 * (1 - attn)), int(255 * (1 - attn)))
        return '<span style="background-color: {}"> {}</span>'.format(html_color, word)

    def mk_html(self, index, batch, preds, normlized_weights_1, normlized_weights_2, TEXT):
        "HTMLデータを作成する"

        # indexの結果を抽出
        sentence = batch.Text[0][index]  # 文章
        label = batch.Label[index]  # ラベル
        pred = preds[index]  # 予測

        # indexのAttentionを抽出と規格化
        attens1 = normlized_weights_1[index, 0, :]  # 0番目の<cls>のAttention
        attens1 /= attens1.max()

        attens2 = normlized_weights_2[index, 0, :]  # 0番目の<cls>のAttention
        attens2 /= attens2.max()

        # ラベルと予測結果を文字に置き換え
        if label == 0:
            label_str = "Negative"
        else:
            label_str = "Positive"

        if pred == 0:
            pred_str = "Negative"
        else:
            pred_str = "Positive"

        # 表示用のHTMLを作成する
        html = '正解ラベル：{}<br>推論ラベル：{}<br><br>'.format(label_str, pred_str)

        # 1段目のAttention
        html += '[TransformerBlockの1段目のAttentionを可視化]<br>'
        for word, attn in zip(sentence, attens1):
            html += self.__highlight(TEXT.vocab.itos[word], attn)
        html += "<br><br>"

        # 2段目のAttention
        html += '[TransformerBlockの2段目のAttentionを可視化]<br>'
        for word, attn in zip(sentence, attens2):
            html += self.__highlight(TEXT.vocab.itos[word], attn)

        html += "<br><br>"

        return html


if __name__ == '__main__':
    instance = Transformer()
    instance.execute()
