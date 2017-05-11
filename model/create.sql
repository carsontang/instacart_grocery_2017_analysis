CREATE TYPE eval_set AS ENUM ('prior', 'train', 'test');

CREATE TABLE orders (
	order_id serial PRIMARY KEY,
	user_id serial,
	order_eval_set eval_set,
	order_number smallserial,
	order_dow smallserial,
	order_hour_of_day smallserial,
	days_since_prior numeric DEFAULT NULL /*  CONSTRAINT positive_days_since_prior CHECK */
);

CREATE TABLE departments (
	department_id serial PRIMARY KEY,
	department text
);

CREATE TABLE aisles (
	aisle_id serial PRIMARY KEY,
	aisle text
);

CREATE TABLE products (
	product_id serial PRIMARY KEY,
	product_name text,
	aisle_id serial REFERENCES aisles(aisle_id),
	department_id serial REFERENCES departments(department_id)
);

CREATE TABLE order_products__train (
	order_id serial references orders(order_id),
	product_id serial references products(product_id),
	add_to_cart_order serial,
	reordered boolean,
	PRIMARY KEY (order_id, product_id)
);

CREATE TABLE order_products__prior (
	order_id serial references orders(order_id),
	product_id serial references products(product_id),
	add_to_cart_order serial,
	reordered boolean,
	PRIMARY KEY (order_id, product_id)
);

/* alter table orders alter column days_since_prior set data type numeric; */
