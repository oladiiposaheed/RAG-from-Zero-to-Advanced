"""
Module 3 – Task 7: Error Handling and Robust Parsing

This script demonstrates:
- Defining a Pydantic model.
- Using normal PydanticOutputParser.
- Using OutputFixingParser to auto-correct invalid output.
"""

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


# Model for a person's basic info
class PersonInfo(BaseModel):
    """Person info model.

    Attributes:
        name: Name of the person.
        age: Age of the person.
        gender: Gender of the person.
    """
    name: str = Field(description='Name of the person')
    age: int = Field(description='Age of the person')
    gender: str = Field(description='Gender of the person')


def create_parsers():
    """Create normal and fixing parsers.

    Returns:
        tuple: (normal_parser, fixing_parser)
    """
    # Normal parser that expects valid JSON
    normal_parser = PydanticOutputParser(pydantic_object=PersonInfo)

    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

    # Parser that auto-corrects malformed output using the LLM
    fixing_parser = OutputFixingParser.from_llm(
        parser=normal_parser,
        llm=llm
    )

    return normal_parser, fixing_parser


def main() -> None:
    """Run error handling example."""
    load_dotenv()

    # Create parsers
    normal_parser, fixing_parser = create_parsers()

    # Bad output that is not valid JSON
    bad_output = 'name Fatimah age 25 gender female'

    # Try normal parser (should fail)
    try:
        result = normal_parser.parse(bad_output)
        print(f'Parsed normally: {result}')
    except Exception as e:
        print('Normal parse error:')
        print(e)

    # Try fixing parser (should succeed after LLM correction)
    try:
        fixed_result = fixing_parser.parse(bad_output)
        print('\nFixed parse result:')
        print(fixed_result)
        print('Type:', type(fixed_result).__name__)
    except Exception as e:
        print('Fixing parser also failed:')
        print(e)


if __name__ == '__main__':
    main()