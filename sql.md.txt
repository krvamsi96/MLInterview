* [What is the difference between base and derived relation? ](#base)
* [What is the difference between primary and foreign key?](#primary)
* [What is an index represent in relational database model?](#indexrepresent)
* [What do you understand by database Normalization?](#Normalization)
* [What are the different types of normalization that exists in the database?](#typesNormalization)
* [How de-normalization is different from normalization?](#de-normalization)
* [What is the type of de-normalization?](#dn)
* [What is the difference between extension and intension?](#ei)
* [What is the difference between DBMS and RDBMS](#RDBMS)
* [What is Index?](#Index)
* [What is Trigger?](#Trigger)
* [What is a NOLOCK?](#NOLOCK) 
* [Differentiate UNION, MINUS, UNION ALL and INTERSECT?](#union)
* [What is Similarity and Difference between Truncate and Delete in SQL?](#td)
* [What are COMMIT and ROLLBACK in SQL? ](#cr)
* [What are Self Join and Cross Join?](#cj)
* [What is a join? Explain the different types of joins?](#dj)
* [What is DDL, DML and DCL?](#dd)
* [ What is Index tuning?](#it)

* [ALTER TABLE](#at)
* [AND](#an)
* [AS](#as)
* [AVG](#av)
* [BETWEEN](#be)
* [COUNT](#co)
* [CREATE TABLE](#ct)
* [DELETE](#de)
* [GROUP BY](#gb)
* [INNER JOIN](#ij)
* [INSERT](#in)
* [LIKE](#like)
* [LIMIT](#li)
* [Max](#ma)
* [MIN](#mi)
* [OR](#or)
* [ORDER BY](#ob)
* [OUTER JOIN](#oj)
* [ROUND](#ro)
* [SELECT](#se)
* [SELECT DISTINCT](#sd)
* [SUM](#su)
* [UPDATE](#up)
* [WHERE](#wh)





## What is the difference between base and derived relation?  <a name="base"></br>
- Relational database means the relationship between different databases. In relational database user can store and access all the data through the tables which are related to each other. 

- Relationship between the store data is called base relations and implementation of it is called as tables. Whereas, relations which don’t store the data, but can be found out by applying relational operations on other relations are called as derived relations. When these are implemented they are termed as views or queries.

- Derived relations are more useful then base relation, as they can have more information from many relations, but they act as a single relation.




## What is the difference between primary and foreign key?  <a name="primary"></br>
- Primary key uniquely identify a relationship in a database, whereas foreign key is the key that is in other relation and it has been referenced from the primary key from other table.
- Primary key remains one only for the table, whereas there can be more than one foreign key.
- Primary key is unique and won’t be shared between many tables, but foreign key will be shared between more than one table and will be used to tell the relationship between them.


## What is an index represent in relational database model?  <a name="indexrepresent"></br> 

- Index is a way to provide quick access to the data and structure. It has indexes maintain and can be created to combine attributes on a relation. Index allows the queries to filter out the searches faster and matching data can be found earlier with simplicity. 

- For example: It is same as the book where by using the index you can directly jump to a defined section. In relational database there is a provision to give multiple indexing techniques to optimize the data distribution.


## What do you understand by database Normalization? <a name="Normalization"></br> 

- Normalization is very essential part of relational model. 
- Normal forms are the common form of normalization. 
- It helps in reducing redundancy to increase the information overall. 
- It has some disadvantages as it increases complexity and have some overhead of processing.
- It consists of set of procedures that eliminates the domains that are non-atomic and redundancy of data that prevents data manipulation and loss of data integrity.

## What are the different types of normalization that exists in the database? <a name="typesNormalization"></br> 

There are 9 normalizations that are used inside the database. These are as follows:
1. First normal form: in this table represents a relation that has no repeating groups.
2. Second normal form: non- prime attributes are not functional dependent on subset of any candidate key.
3. Third normal form: in a table every non- prime attribute is non-transitively dependent on every candidate key
4. Elementary key normal form: superkey dependency or elementary key dependency effects the functional dependency in a table.
5. Boyce codd normal form: “every non-trivial functional dependency in the table is dependent on superkey”.
6. Fourth normal form: “Every non-trivial multivalued dependency in the table is a dependent on a superkey”.
7. Fifth normal form (5NF): “Every non-trivial join dependency in the table is implied by the superkeys of the table”.
8. Domain/key normal form (DKNF): “Every constraint on the table is a logical consequence of the table's domain constraints and key constraints”.
9. Sixth normal form (6NF): “Table features no non-trivial join dependencies at all”.

## How de-normalization is different from normalization? <a name="de-normalization"></br> 

- Analytical processing databases are not very normalized. The operations which are used are read most databases. 
- It is used to extract the data that are ancient and accumulated over long period of time. For this purpose de-normalization occurs that provide smart business applications.
- Dimensional tables in star schema are good example of de-normalized data. 
- The de-normalized form must be controlled while extracting, transforming, loading and processing. 
- There should be constraint that user should not be allowed to view the state till it is consistent.
- It is used to increase the performance on many systems without RDBMS platform.

## What is the type of de-normalization? <a name="dn"></br> 

Non-first normal form (NFA) 

– It describes the definition of the database design which is different from the first normal form.
- It keeps the values in structured and specialized types with their own domain specific languages. 
- The query language used in this is extended to incorporate more support for relational domain values by adding more operators.


## What is the difference between extension and intension?  <a name="ei"></br> 

The major difference between extension and intension is that:

- Extension is time dependent, whereas intension includes a constant value.
- Extension tells about the number of tuples presented in a table at any instance, whereas intension gives the name, structure and constraint of the table.

## What is the difference between DBMS and RDBMS? <a name="RDBMS"></br> 

- DBMS is persistent and accessible when the data is created or exists, but RDBMS tells about the relation between the table and other tables.
- RDBS supports a tabular structure for data and relationship between them in the system whereas DBMS supports only the tabular structure.
- DBMS provide uniform methods for application that has to be independently accessed, but RDBMS doesn’t provide methods like DBMS but provide relationship which link one entity with another.

---------------------------------------------------

## What is Index? <a name="Index"></br> 

- A pointer to data having physical representation is called as Index.
- Record can be located quickly and efficiently by creating Indices on existing tables.
- Each index in a table has some valid name and we can have more than one index in different columns of a table.
- We can speed up queries by setting up index in a column of a table.
- In a table , each row is examined by sql server to fulfil our query is known as table scan and it only happen when there is no index available to help the query.
- On large tables, the table scan has huge impact on performance.
- Clustered and Non clustered indexes are the most widely used indexes in a database.

## What is Trigger?  <a name="Trigger"></br> 

- A Trigger is a process of firing an action when some event like Insert, Update or Delete occurs.
- A trigger can’t be called or even executed rather they are automatically become active by the DBMS whenever some modification in associated table occur.
- Triggers are event driven and can attached to particular table in a database.
- Triggers are implicitly executed and stored procedures are also executed by triggers.
- Referential integrity is maintained by the trigger and they are managed and stored by DBMS.
- Triggers can be nested also, in which Insert, Update or Delete logic can be fired from the trigger itself.

## What is a NOLOCK? <a name="NOLOCK"></br> 

- NOLOCK is used to improve concurrency on a busy system.
- On data read, no lock can be taken on SELECT statement.
- When some other process is updating the data on the same time you are reading it is known as dirty read.
- Read (Shared) locks are taken by SELECT Statements.
- Simultaneous access of multiple SELECT statements is allowed in Shared lock but modification process is not allowed.
- The result to your system is blocking.
- Update will start on completion of all the reads.


## What is Similarity and Difference between Truncate and Delete in SQL? <a name="td"></br> 

- Similarity

- Both Truncate and Delete command will delete data from given table and they will not delete the table structure from the database.

- Difference

1. TRUNCATE is a DDL (data definition language) command whereas DELETE is a DML (data manipulation language) command.

2. We can’t execute a trigger with TRUNCATE whereas with DELETE command, a trigger can be executed.

3. We can use any condition in WHERE clause using DELETE but it is not possible with TRUNCATE.

4. If table is referenced by any foreign key constraints then TRUNCATE cannot work.

5. TRUNCATE is faster than DELETE, because when you use DELETE to delete the data, at that time it store the whole data in rollback space from where you can get the data back after deletion, whereas TRUNCATE will not store data in rollback space and will directly delete it. You can’t get the deleted data back when you use TRUNCATE.

## What are COMMIT and ROLLBACK in SQL? <a name="cr"></br> 

- COMMIT statement is used to end the current transaction and once the COMMIT statement is exceucted the transaction will be permanent and undone.
- Syntax: COMMIT;
- Example:
```
BEGIN
UPDATE EmpDetails SET EmpName = ‘Arpit’ where Dept = ‘Developer’
COMMIT;
END;
```

-ROLLBACK statement is used to end the current transaction and undone the changes which was made by that transaction.
- Syntax: ROLLBACK [TO] Savepoint_name;
- Example:
```
BEGIN
Statement1;
SAVEPOINT mysavepoint;
BEGIN
Statement2;
EXCEPTION
WHEN OTHERS THEN
ROLLBACK TO mysavepoint;
Statement5;
END;
END;
```

## What is a WITH(NOLOCK)?  

- WITH(NOLOCK) is used to unlock the data which is locked by the transaction that is not yet committed. This command is used before SELECT statement.
- When the transaction is committed or rolled back then there is no need to use NOLOCK function because the data is already released by the committed transaction.
- Syntax: WITH(NOLOCK)
- Example:
```
SELECT * FROM EmpDetails WITH(NOLOCK)
WITH(NOLCOK) is similar as READ UNCOMMITTED.
```

## Differentiate UNION, MINUS, UNION ALL and INTERSECT? <a name="union"></br> 

- INTERSECT - It will give all the distinct rows from both select queries.
- MINUS - It will give distinct rows returned by the first query but not by the second query.
- UNION - It will give all distinct rows selected by either first query or second query.
- UNION ALL - It will give all rows returned by either query with all duplicate records.

## What are Self Join and Cross Join?  <a name="cj"></br> 

- When we want to join a table to itself then SELF JOIN is used.
- We can give one or more aliases to eliminate the confusion.
- A self join can be used as any type, if both the tables are same.
- The simple example where we can use SELF JOIN is if in a company have a hierarchal reporting structure and an employee reports to another.
- A cross join give the number of rows in the first table multiplied by the number of rows in second table.
- The simple example where we can use CROSS JOIJ is if in an organization wants to combine every Employee with family table to see each Employee with each family member.

## What is a join? Explain the different types of joins?    <a name="dj"></br> 

Using Join in a query, we can retrieve referenced columns or rows from multiple tables.

Following are different types of Joins:

1. JOIN: Return details from tables if there is at least one matching row in both tables.
2. LEFT JOIN: It will return all rows from the left table, even if there are no matching row in the right table.
3. RIGHT JOIN: It will return all rows from the right table, even if there is no matching row in the left table.
4. FULL JOIN: It will return rows when there is a match in either of tables.

## What is DDL, DML and DCL?  <a name="dd"></br> 

SQL commands can be divided in three large subgroups.

1) DDL: The SQL commands which deals with database schemas and information of how the data will be generated in database are classified as Data Definition Language.
-For example: CREATE TABLE or ALTER TABLE belongs to DDL.

2) DML: The SQL commands which deals with data manipulation are classified as Data Manipulation Language.
For example: SELECT, INSERT, etc.

3) DCL: The SQL commands which deal with rights and permission over the database are classified as DCL.
For example: GRANT, REVOKE


## What is Index tuning? <a name="it"></br> 

Index tuning is part of database tuning for selecting and creating indexes. The index tuning goal is to reduce the query processing time. Potential use of indexes in dynamic environments with several ad-hoc queries in advance is a difficult task. Index tuning involves the queries based on indexes and the indexes are created automatically on-the-fly. No explicit actions are needed by the database users for index tuning.




---------------------------------------------------
---------------------------------------------------

## MySQL Basics 


### ALTER TABLE <a name="at"></br> 

```
ALTER TABLE table_name ADD column datatype;
```

`ALTER TABLE` lets you add columns to a table in a database.

### AND <a name="an"></br> 

```
SELECT column_name(s)
FROM table_name
WHERE column_1 = value_1
AND column_2 = value_2;
```

`AND` is an operator that combines two conditions. Both conditions must be true for the row to be included in the result set.

### AS <a name="as"></br> 

```
SELECT column_name AS 'Alias'
FROM table_name;
```

`AS` is a keyword in SQL that allows you to rename a column or table using an *alias*.

### AVG <a name="av"></br> 

```
SELECT AVG(column_name)
FROM table_name;

```

`AVG()` is an aggregate function that returns the average value for a numeric column.

### BETWEEN <a name="be"></br> 

```
SELECT column_name(s)
FROM table_name
WHERE column_name BETWEEN value_1 AND value_2;
```

The `BETWEEN` operator is used to filter the result set within a certain range. The values can be numbers, text or dates.

### COUNT <a name="co"></br> 

```
SELECT COUNT(column_name)
FROM table_name;
```

`COUNT()` is a function that takes the name of a column as an argument and counts the number of rows where the column is not `NULL`.

### CREATE TABLE <a name="ct"></br> 

```
CREATE TABLE table_name (column_1 datatype, column_2 datatype, column_3 datatype);
```

`CREATE TABLE` creates a new table in the database. It allows you to specify the name of the table and the name of each column in the table.

### DELETE <a name="de"></br> 

```
DELETE FROM table_name WHERE some_column = some_value;
```

`DELETE` statements are used to remove rows from a table.

### GROUP BY <a name="gb"></br> 

```
SELECT COUNT(*)
FROM table_name
GROUP BY column_name;
```

`GROUP BY` is a clause in SQL that is only used with aggregate functions. It is used in collaboration with the `SELECT` statement to arrange identical data into groups.

### INNER JOIN <a name="ij"></br> 

```
SELECT column_name(s) FROM table_1
JOIN table_2
ON table_1.column_name = table_2.column_name;
```

An inner join will combine rows from different tables if the *join condition* is true.

### INSERT <a name="in"></br> 

```
INSERT INTO table_name (column_1, column_2, column_3) VALUES (value_1, 'value_2', value_3);
```

`INSERT` statements are used to add a new row to a table.

### LIKE <a name="like"></br> 

```
SELECT column_name(s)
FROM table_name
WHERE column_name LIKE pattern;
```

`LIKE` is a special operator used with the `WHERE` clause to search for a specific pattern in a column.

### LIMIT <a name="li"></br> 

```
SELECT column_name(s)
FROM table_name
LIMIT number;
```

`LIMIT` is a clause that lets you specify the maximum number of rows the result set will have.

### MAX <a name="ma"></br> 

```
SELECT MAX(column_name)
FROM table_name;
```

`MAX()` is a function that takes the name of a column as an argument and returns the largest value in that column.

### MIN <a name="mi"></br> 

```
SELECT MIN(column_name)
FROM table_name;
```

`MIN()` is a function that takes the name of a column as an argument and returns the smallest value in that column.

### OR <a name="or"></br> 

```
SELECT column_name
FROM table_name
WHERE column_name = value_1
OR column_name = value_2;
```

`OR` is an operator that filters the result set to only include rows where either condition is true.

### ORDER BY <a name="ob"></br> 

```
SELECT column_name
FROM table_name
ORDER BY column_name ASC|DESC;
```

`ORDER BY` is a clause that indicates you want to sort the result set by a particular column either alphabetically or numerically.

### OUTER JOIN <a name="oj"></br> 

```
SELECT column_name(s) FROM table_1
LEFT JOIN table_2
ON table_1.column_name = table_2.column_name;
```

An outer join will combine rows from different tables even if the the join condition is not met. Every row in the *left* table is returned in the result set, and if the join condition is not met, then `NULL` values are used to fill in the columns from the *right* table.

### ROUND <a name="ro"></br> 

```
SELECT ROUND(column_name, integer)
FROM table_name;
```

`ROUND()` is a function that takes a column name and an integer as an argument. It rounds the values in the column to the number of decimal places specified by the integer.

### SELECT <a name="se"></br> 

```
SELECT column_name FROM table_name;
```

`SELECT` statements are used to fetch data from a database. Every query will begin with SELECT.

### SELECT DISTINCT <a name="sd"></br> 

```
SELECT DISTINCT column_name FROM table_name;
```

`SELECT DISTINCT` specifies that the statement is going to be a query that returns unique values in the specified column(s).

### SUM <a name="su"></br> 

```
SELECT SUM(column_name)
FROM table_name;
```

`SUM()` is a function that takes the name of a column as an argument and returns the sum of all the values in that column.

### UPDATE <a name="up"></br> 

```
UPDATE table_name
SET some_column = some_value
WHERE some_column = some_value;
```

`UPDATE` statments allow you to edit rows in a table.

### WHERE <a name="wh"></br> 

```
SELECT column_name(s)
FROM table_name
WHERE column_name operator value;
```

`WHERE` is a clause that indicates you want to filter the result set to include only rows where the following *condition* is true.
