'''
    Pydantic Models and Validation
    This script defines a simple Pydantic model for a Person and demonstrates:
    - creating valid instances
    - automatic type conversion
    - catching validation errors
'''

from pydantic import BaseModel, ValidationError

class Person(BaseModel):
    '''
        A simple model representing a person.

        Attributes:
            name (str): The person's name.
            age (int): The person's age in years.
    '''
    
    name: str
    age: int
    
def create_person(name: str, age: int | str) -> Person:
        '''
        Create a Person instance with automatic validation and conversion.
        '''
        
        return Person(name=name, age=age)
    
def main() -> None:
    '''
    Run examples demonstrating Pydantic validation.
    '''
    
    # Example 1: Valid person
    person1 = create_person('Aisha', 22)
    print(f'Person1: {person1}')
    print(f'Age type: {type(person1.age).__name__}')
    
    # Automatic conversion from str to int
    person2 = create_person('Favor', '34')
    print(f'\nPerson2: {person2}')
    print(f'Age type: {type(person2.age).__name__}')
    
    # Invalid age
    try: 
        person3 = create_person('Chika', 'forty four')
        print(f'\nPerson3: {person3}')
    
    except Exception as e:
        print('\nValidation failed as expected:')
        print(e)
    
    
if __name__=='__main__':
    main()
        