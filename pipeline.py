import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from postprocessing import FastTextPostProcessor

class TextSimplificationPipeline:
    def __init__(self,
                 model_path="t5_simplification_model",
                 preprocessor=None,
                 freq_threshold=20000):
        """
        model_path: folderul unde ai salvat modelul T5 antrenat
        preprocessor: instanță de TextPreprocessor cu fastText încărcat
        freq_threshold: prag pentru cuvintele 'complexe'
        """
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.post_processor = None
        if preprocessor:
            self.post_processor = FastTextPostProcessor(preprocessor, frequency_threshold=freq_threshold)

    def simplify_with_t5(self, text: str, max_length=128, num_beams=4) -> str:
        """
        Simplifică textul folosind modelul T5.
        """
        # Pregătim input
        input_text = "simplify: " + text
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")

        # Generăm
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
        simplified_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return simplified_text

    def post_process(self, text: str) -> str:
        """
        Înlocuiește cuvintele complexe cu sinonime (folosind fastText).
        """
        if self.post_processor:
            return self.post_processor.process_text(text)
        else:
            return text

    def simplify(self, text: str, apply_postprocess=True) -> str:
        """
        Pipeline complet: (1) T5 -> (2) post-procesare fastText (opțional).
        """
        # 1. T5 simplification
        simplified_t5 = self.simplify_with_t5(text)

        # 2. Post-processing
        if apply_postprocess:
            final_text = self.post_process(simplified_t5)
        else:
            final_text = simplified_t5

        return final_text
