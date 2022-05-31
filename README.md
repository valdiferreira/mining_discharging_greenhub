# Mineração de dados de crowdsourcing para investigar o uso de energia em dispositivos Android

Utilização de técnicas tradicionais de aprendizado de máquina para prever o uso de bateria de dispositivos móveis a partir de dados de crowdsourcing provenientes do GreenHub.

## 🔧 Scripts

### Função 01:
- Seleção dos dados de descarga da bateria
- Agrupamento dos processos
- Agregagação dos processos aos samples
- Criação da variável alvo
     consume_time
- Criação das variáveis independentes:
     user_storage_busy_percent
     system_size_partition_busy
     ram_busy_percent

### Função 02:
- Seleciona as features de interesse para processamento

### Função 03:
- Separação dos dados em arquivos por modelo

### Função 04:
- Estimativa de Outliers (Método do Boxplot Ajustado)

### Função 05:
- Tunning dos meta-parâmetros dos algortimos

### Função 06:
- Realização e avaliação das predições
- Estimativa do impacto das variáveis independente (fatores) nas predições

## 🤝 Contribuição

Projeto aberto para ajuda!

