from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


class QueryVariations(BaseModel):
    variations: list[str] = Field(
        ...,
        description="A list of diverse query variations for information retrieval.",
        min_items=3,
    )

class QueryRewriter:
    """
    Reescrever queries usando LLM.
    """

    def __init__(self, model: str = "gpt-4o-mini", n: int = 3):
        self.n = n
        self.llm = ChatOpenAI(model=model, temperature=0.2)
        self.parser = PydanticOutputParser(pydantic_object=QueryVariations)
        self.prompt = PromptTemplate(
            template=(
                "Você reescreve consultas para melhorar a recuperação em um documento oficial (BNCC).\n"
                "Não invente fatos. Use português.\n\n"
                "Consulta original:\n{query}\n\n"
                "Gere exatamente {n} variações curtas, estilo consulta de busca.\n"
                "- Preserve restrições (ano/série/tema) se existirem.\n"
                "- Foque em termos como: habilidades, competências, objetos de conhecimento, unidade temática.\n\n"
                "- Tente diversificar se o conteúdo for amplo.\n"
                "{format_instructions}\n"
            ),
            input_variables=["query"],
            partial_variables={
                "n": str(n),
                "format_instructions": self.parser.get_format_instructions(),
            },
        )
        self.chain = self.prompt | self.llm | self.parser

    def rewrite(self, query: str) -> list[str]:
        result: QueryVariations = self.chain.invoke({"query": query})
        variations = result.variations

        return variations[: self.n]
    
