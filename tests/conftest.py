"""Pytest 共享 fixtures。"""

import pytest


@pytest.fixture
def sample_medicine_text() -> str:
    """医学论文样本。"""
    return (
        "This study investigates the efficacy of a novel therapeutic approach "
        "for treating chronic inflammatory diseases. We conducted a randomized "
        "clinical trial involving 500 patients diagnosed with rheumatoid arthritis. "
        "The treatment group received a combination of methotrexate and biologic "
        "agents, while the control group received methotrexate alone. "
        "Results demonstrate a significant reduction in inflammatory markers "
        "including CRP and ESR levels in the treatment group (p < 0.001). "
        "Furthermore, patients reported improved quality of life scores. "
        "These findings suggest that combination therapy may be superior to "
        "monotherapy for managing chronic inflammatory conditions."
    )


@pytest.fixture
def sample_cs_text() -> str:
    """计算机科学论文样本。"""
    return (
        "We propose a novel deep learning architecture for semantic segmentation "
        "of medical images. Our approach utilizes a transformer-based encoder "
        "combined with a lightweight decoder to achieve state-of-the-art performance "
        "while maintaining computational efficiency. The model is trained on a "
        "dataset of 10,000 annotated CT scans. Experimental results show that "
        "our method achieves 92.3% mIoU on the test set, outperforming existing "
        "approaches by 3.5%. Additionally, we introduce a novel attention mechanism "
        "that captures long-range dependencies more effectively than traditional "
        "convolutional methods. The code and pre-trained models are publicly available."
    )


@pytest.fixture
def sample_humanities_text() -> str:
    """人文社科论文样本。"""
    return (
        "This paper examines the sociocultural impact of digital transformation "
        "on indigenous communities in Southeast Asia. Drawing on ethnographic "
        "fieldwork conducted over eighteen months, I argue that technological "
        "adoption creates complex negotiations between tradition and modernity. "
        "The findings reveal that while digital tools enable economic participation, "
        "they also challenge existing social hierarchies and knowledge transmission "
        "systems. This research contributes to broader debates about technological "
        "determinism and cultural resilience. In conclusion, I suggest that "
        "policymakers should adopt culturally sensitive approaches to digital "
        "infrastructure development in indigenous contexts."
    )


@pytest.fixture
def sample_ai_generated_text() -> str:
    """AI生成文本样本（模仿学术论文风格）。"""
    return (
        "Artificial intelligence has revolutionized many fields in recent years. "
        "The applications of machine learning are vast and diverse. Deep learning "
        "models have achieved remarkable results in various tasks. Natural language "
        "processing has seen significant advances with transformer architectures. "
        "Computer vision has also benefited from convolutional neural networks. "
        "Reinforcement learning has enabled breakthroughs in game playing and robotics. "
        "The future of AI research looks promising with many opportunities for innovation. "
        "Researchers continue to explore new algorithms and architectures. "
        "The impact of AI on society will be profound and far-reaching. "
        "It is important to consider the ethical implications of AI development."
    )


@pytest.fixture
def sample_long_text() -> str:
    """较长文本用于测试。"""
    paragraphs = []
    for i in range(10):
        paragraphs.append(
            f"Paragraph {i + 1}. This is a test paragraph with multiple sentences. "
            "It contains enough content to test various features of the system. "
            "We need sufficient length to ensure meaningful analysis results. "
            "The quick brown fox jumps over the lazy dog near the riverbank. "
            "Scientific research requires careful methodology and rigorous testing. "
            "Data analysis must be performed with appropriate statistical methods. "
            "Results should be reproducible by independent research groups. "
            "Peer review is an essential component of the scientific process. "
        )
    return "\n\n".join(paragraphs)
