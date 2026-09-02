'''
Field Descriptions and Defaults
'''


from pydantic import BaseModel, Field, ValidationError

class Product(BaseModel):
    '''A model representing a product.'''
    
    name: str = Field(description='Name of the product', min_length=2)
    price: float = Field(description='Price in Naira', gt=0)
    category: str = Field(description='Product category', default='general')
    
    
def create_product(name: str, price: float, category: str = 'general') -> Product:
    '''Create a validated Product instance.'''
    
    return Product(name=name, price=price, category=category)


def main() -> None:
    
    # Valid product with default category
    product1 = create_product('Rice', '20000.99')
    print(f'Product1: {product1}')
    print(f'Category: {product1.category}')
    
    # Valid product with custom category
    product2 = create_product('Laptop', 600000, category='electronics')
    print(f'\nProduct 2: {product2}')
    
    # Invalid name (too short)
    try:
        product3 = create_product('M', 100)
        print(f'\nProduct3: {product3}')
    except ValidationError as e:
        print('\nValidation failed for name:')
        print(e)
    
    # Valid product with custom category
    try:
        product4 = create_product('Bread', -10)
        print(f'\nProduct 4{product4}')
    
    except Exception as e:
        print(e)
        

if __name__=='__main__':
    main()