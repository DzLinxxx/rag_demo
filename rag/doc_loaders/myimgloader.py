from typing import List

from langchain_community.document_loaders.unstructured import UnstructuredFileLoader


# from doc_loaders.ocr import get_ocr
def get_ocr(use_cuda: bool = True):
    try:
        from rapidocr_paddle import RapidOCR

        ocr = RapidOCR(
            det_use_cuda=use_cuda, cls_use_cuda=use_cuda, rec_use_cuda=use_cuda
        )
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR

        ocr = RapidOCR()

    return ocr

class RapidOCRLoader(UnstructuredFileLoader):
    def _get_elements(self) -> List:
        def img2text(filepath):
            resp = ""
            ocr = get_ocr()
            result, _ = ocr(filepath)
            if result:
                ocr_result = [line[1] for line in result]
                resp += "\n".join(ocr_result)
            return resp

        text = img2text(self.file_path)
        from unstructured.partition.text import partition_text

        return partition_text(text=text, **self.unstructured_kwargs)


if __name__ == "__main__":
    loader = RapidOCRLoader(file_path=r"C:\Users\19652\Desktop\test.png")
    docs = loader.load()
    print(docs)
