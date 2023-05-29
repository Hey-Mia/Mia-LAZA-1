![Logo](https://i.ibb.co/jW2dV9W/Kopie-n-vrhu-LAZA-1-removebg-preview.png)


# LAZA - 1

LAZA - 1 is the first model from Heexy, it's an open-source testing AI, the goal is to make this AI able to interact with people and help them, the goal is to make this an LLM.
## Installation

```bash
pip install -r requirements.txt
```

## Other commands

#### Run the AI (The AI starts to learn from the dataset and after it learns it turns on the chatbot function in the terminal)
```bash
python index.py
```
**Note: ⚠️ The process can be very demanding on CPU + RAM**¨

#### Run chat with AI
```bash
python chat.py
```

#### Run learning of model
```bash
python chat.py
```

#### Generate new requirements.txt
```bash
pipreqs . --force
```

## How is works?

This AI model of **LAZA-1** works based on a knowledge transfer technique using the **T5 (Text-to-Text Transfer Transformer)** model. The T5 model is a transformer model that is trained on large text data and then adapted for a specific task, in this case answering questions.

The **LAZA-1** model is trained on a large amount of text and has the ability to understand questions and generate answers based on the learned knowledge. In training the **LAZA-1** model, fine-tuning is used where a pre-trained T5 model is loaded and modified for the specific task of answering questions.

To generate the question answers, the input question is first preprocessed using a tokenizer that splits the question into tokens and converts them into a numerical representation. Then, the input question is passed to the **LAZA-1** model, which generates the answer based on the learned patterns and knowledge.

**LAZA-1** is capable of generating answers to various questions and can be used in different domains such as customer support, chatbots, information retrieval and more. Its success rate and quality of answers depends on the quality of the training data and the extent of knowledge learned.
###
![T5 Example](https://1.bp.blogspot.com/-89OY3FjN0N0/XlQl4PEYGsI/AAAAAAAAFW4/knj8HFuo48cUFlwCHuU5feQ7yxfsewcAwCLcBGAsYHQ/s1600/image2.png)

![T5 Example](https://miro.medium.com/max/4006/1*D0J1gNQf8vrrUpKeyD8wPA.png)

## Support

For support, email info@heexy.org or join our [Discord](https://discord.gg/uWUQKsm2HU) community.

## Authors

- [@WhaTeery1087 - Backend & Front-end](https://github.com/Whtery1087)