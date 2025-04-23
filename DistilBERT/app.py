import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertConfig
import torch

# Set title
st.title("Hate Speech Detector")

MODEL_PATH = "best.pt"
ID2LABEL = {0: "hate_speech", 1: "normal", 2: "offensive"}
LABEL2ID = {label: idx for idx, label in ID2LABEL.items()}
NUM_LABELS = len(ID2LABEL)
MODEL_NAME = "bert-base-uncased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_student_model(teacher_model):
    student_config = teacher_model.config.to_dict()
    student_config['num_hidden_layers'] //= 2  # half the number of hidden layer
    student_config = BertConfig.from_dict(student_config)
    model = type(teacher_model)(student_config)
    return model


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    teacher_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    student_model = build_student_model(teacher_model)
    student_model.load_state_dict(
        torch.load("best.pt", map_location=DEVICE))
    return tokenizer, student_model


user_input = st.text_area("Enter your text here:", height=120)

if st.button("Submit"):
    if not user_input.strip():
        st.info("Waiting for user input...")

    status = st.status("Starting...", expanded=True)

    try:
        status.update(label="Loading Model...", state="complete")
        tokenizer, model = load_model()
        model.eval()
        model.to(DEVICE)

        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            pred_label = torch.argmax(probs, dim=1).item()

        status.update(label="Finished", state="complete")
        st.subheader(f"Prediction: {ID2LABEL[pred_label]}")
        for idx, score in enumerate(probs[0]):
            st.write(f"{ID2LABEL[idx]}: **{score:.2%}**")

    except Exception as e:
        status.update(label=f"Error：{e}", state="error")