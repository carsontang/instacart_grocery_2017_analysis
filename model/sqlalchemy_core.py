# Connect to the database
import sqlalchemy
from sqlalchemy import create_engine

engine = create_engine('postgresql+psycopg2://ctang:@localhost:5432/ctang', encoding='utf-8') # no password
connection = engine.connect()

# Set up the tables
from sqlalchemy import MetaData
from sqlalchemy import Table, Column, Enum, Boolean, Integer, Numeric, Text

metadata = MetaData()

orders = Table('orders', metadata,
    Column('order_id', Integer(), autoincrement=True, primary_key=True),
    Column('user_id', Integer(), autoincrement=True),
    Column('order_eval_set', Enum('prior', 'train', 'test')),
    Column('order_number', Integer()),
    Column('order_dow', Integer()),
    Column('order_hour_of_day', Integer()),
    Column('days_since_prior', Numeric(), nullable=True)
)

aisles = Table('aisles', metadata,
    Column('aisle_id', Integer(), autoincrement=True, primary_key=True),
    Column('aisle', Text)
)

departments = Table('departments', metadata,
    Column('department_id', Integer(), autoincrement=True, primary_key=True),
    Column('department', Text)
)

products = Table('products', metadata,
    Column('product_id', Integer(), autoincrement=True, primary_key=True),
    Column('product_name', Text()),
    Column('aisle_id', ForeignKey('aisles.aisle_id')),
    Column('department_id', ForeignKey('departments.department_id'))
)

order_products__train = Table('order_products__train', metadata,
    Column('order_id', ForeignKey('orders.order_id')),
    Column('product_id', ForeignKey('products.product_id')),
    Column('add_to_cart_order', Integer(), autoincrement=True),
    Column('reordered', Boolean())
)

from sqlalchemy.sql import select, func, desc

# Top 10 departments with the most product listings
columns = [departments.c.department, func.count(products.c.product_id).label('product_count')]
s = select(columns) \
    .select_from(departments.join(products)) \
    .group_by(departments.c.department) \
    .order_by(desc('product_count'))

rp = connection.execute(s)
print "%20s | %s" % ('department', 'number of products')
print '-' * 50
for record in rp:
    print "%20s | %s" % (record.department, record.product_count)

# /* Top 10 most ordered items */
columns = [order_products__train.c.product_id, func.count().label('num_ordered')]
product_counts = select(columns) \
    .select_from(order_products__train) \
    .group_by(order_products__train.c.product_id) \
    .order_by(desc('num_ordered')) \
    .limit(10) \
    .subquery('product_counts')

col2 = [products.c.product_name, product_counts.c.num_ordered]
r = select(col2).select_from(product_counts).join(products).limit(10)
print str(product_counts)
rp = connection.execute(s)
for record in rp:
    print record

    # SELECT p.product_name, o.num_ordered, a.aisle
    #   FROM (SELECT o.product_id as product_id, COUNT(*) AS num_ordered
    #           FROM order_products__train o
    #       GROUP BY o.product_id
    #       ORDER BY num_ordered DESC) AS o
    #   JOIN products p ON p.product_id = o.product_id
    #   JOIN aisles a ON p.aisle_id = a.aisle_id
    #  LIMIT 10;