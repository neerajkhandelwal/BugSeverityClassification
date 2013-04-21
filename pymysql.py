import mysql.connector
from mysql.connector import errorcode

class DB:
	config = {}
	conn = ''
	cursor = ''
	
	def __init__(self, host='localhost', user='root', password='', db='test'): 
		self.config = {'host': host, 'user': user, 'password': password, 'db': db} 
		try:
			conn = mysql.connector.connect(**self.config)
			self.conn = conn
			self.cursor = conn.cursor()
			print "Connected to database: %s." % self.config['db']
			
		
		except mysql.connector.Error as err:
			if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
				print "Wrong username or password!"
			elif err.errno == errorcode.ER_BAD_DB_ERROR:
				print "Database %s does not exist" % self.config['db']
			else:
				print err
		
	def close(self):
		self.cursor.close()
		self.conn.close()
		print "Connection closed."