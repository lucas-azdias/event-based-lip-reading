# Rede de Percepção Multigranular de Características Espaço-Temporais (MSTP) para leitura labial baseada em eventos
## Resumo
Este trabalho propõe e avalia um sistema de leitura labial baseado em redes neurais para operação em hardwares limitados, com o objetivo de promover acessibilidade para pessoas com deficiência auditiva. A solução utiliza a arquitetura Multi-grained Spatio-Temporal Features Perceived Network (MSTP) aplicada a dados de eventos do conjunto DVS-Lip, explorando métodos de compressão como destilação de conhecimento e poda de redes neurais. Foram conduzidos experimentos comparativos com versões completas e simplificadas da arquitetura, avaliando acurácia, tempo de inferência, uso de memória e tamanho da rede. Os resultados demonstram que a destilação de conhecimento é capaz de mitigar parcialmente perdas de desempenho em modelos simplificados, enquanto a poda de redes contribui significativamente para a redução do tempo de inferência. No entanto, as limitações observadas na redução do tamanho da rede e a dependência de um único conjunto de dados indicam a necessidade de estudos adicionais para garantir a viabilidade em ambientes reais de hardware embarcado. São discutidos caminhos futuros envolvendo novas arquiteturas, protocolos de validação mais robustos e avaliação em cenários mais diversos.

## Abstract
This work proposes and evaluates a neural network-based lip-reading system designed to operate on resource-constrained hardware, with the goal of enhancing accessibility for individuals with hearing impairments. The solution employs the Multi-grained Spatio-Temporal Features Perceived Network (MSTP) architecture applied to event-based data from the DVS-Lip dataset, incorporating compression techniques such as knowledge distillation and network pruning. Comparative experiments were conducted using both full and simplified versions of the architecture, evaluating accuracy, inference time, memory usage, and model size. The results show that knowledge distillation can partially mitigate performance degradation in simplified models, while network pruning significantly reduces inference time. However, the limited reduction in model size and reliance on a single dataset highlight the need for further studies to ensure practical deployment on embedded hardware. Future work directions include exploring alternative architectures, implementing more robust validation protocols, and evaluating performance in more diverse real-world scenarios.

## Dependências
* Python 3.13
* Pytorch 2.7.0

## Preparando ambiente...
1. Baixe o *dataset* [DVS-Lip](https://drive.google.com/file/d/1dBEgtmctTTWJlWnuWxFtk8gfOdVVpkQ0/view), e insira na pasta `data`;
2. Baixe algum modelo pré-treinado em [modelos](https://drive.google.com/drive/folders/1xi9qoQ0LjEoo6SvWOH2pSXrdjia9_jJC?usp=sharing), e insira na pasta `log`.

## Treinamento
```
python main.py --gpus=0,1 --num_bins=1+4 --test=False --alpha=4 --beta=4 --batch_size=16 --num_workers=8 --log_dir=NOME_MODELO [--distillation] [--teacher_weights=NOME_PROFESSOR] [--prune] [--use_simple]
```

Outros exemplos de comandos podem ser encontrados em `run.sh`.

## Teste
```
python main.py --gpus=0,1 --num_bins=1+4 --test=True --alpha=4 --beta=4 --batch_size=16 --num_workers=8 --reps=5 --weights=NOME_MODELO [--use_simple] [--use_profiler]
```

Outros exemplos de comandos podem ser encontrados em `run.sh`.

## Plotings
Na pasta `plotings` pode ser encontrado exemplos de códigos para plotagem dos dados.

## Referência
Códigos baseados nos providos por [Tan et al. (2022)](https://github.com/tgc1997/event-based-lip-reading)