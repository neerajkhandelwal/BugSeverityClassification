# from pymysql import DB

class Data:
	bugs = []
	bugs_status = {}
	bugs_os = {}
	bugs_priority = {}
	bugs_platform = {}
	bugs_resolution = {}

	def getData(self, limit, wrt_feature='bugs'):
		# bug_id => For identifying the bug.
		
		# bug_severity => For learning.
		# 		Types: blocker, critical, major, normal, minor, trivial, enhancement.

		# bug_status => To get the data which is confirmed.
		# 		Types: UNCONFIRMED, NEW, ASSIGNED, REOPENED, RESOLVED, VERIFIED, CLOSED, READY

		# short_desc => To get the feature for learning.

		# op_sys => May be needed for better visualization of results and testing.
		# 		Types: 50 different

		# priority => May be needed for better visualtization of results and testing. Directly related to severity.
		# 		Types: P5, P4, P3, P2, P1, --

		# rep_platform => May be needed for better visualization of results and testing.
		# 		Types: ____, Sun, x86, Other, PowerPC, All, SGI, DEC, HP, c86_64, ARM, XScale

		# resolution => May be related to severity.
		# 		Types: DUPLICATE, EXPIRED, FIXED, INCOMPLETE, INVALID, MOVED, WONTFIX, WORKSFORME

		query = ("SELECT bug_id, bug_severity, bug_status, short_desc, op_sys, priority, rep_platform, resolution, votes FROM bugs WHERE bug_status != 'NEW' OR bug_status != 'UNCONFIRMED' ORDER BY bug_id ASC LIMIT 0, " + str(limit))
		self.connection.cursor.execute(query)
		for (bug_id, bug_severity, bug_status, short_desc, op_sys, priority, rep_platform, resolution, votes) in self.connection.cursor:
			if bug_id != None and short_desc != None:
				bug = {
					'bug_id': bug_id,
					'severity': bug_severity,
					'status': bug_status,
					'description': short_desc,
					'os': op_sys,
					'priority': priority,
					'platform': rep_platform,
					'resolution': resolution,
					'votes': votes
				}
				self.bugs.append(bug)

				if bug_status != None:
					if bug_status not in self.bugs_status:
						self.bugs_status[bug_status] = []
					self.bugs_status[bug_status].append(bug)

				if op_sys != None:
					if op_sys not in self.bugs_os:
						self.bugs_os[op_sys] = []
					self.bugs_os[op_sys].append(bug)

				if priority != '--' and priority != None:
					if priority not in self.bugs_priority:
						self.bugs_priority[priority] = []
					self.bugs_priority[priority].append(bug)

				if rep_platform != "" and rep_platform != None:
					if rep_platform not in self.bugs_platform:
						self.bugs_platform[rep_platform] = []
					self.bugs_platform[rep_platform].append(bug)

				if resolution != None:
					if resolution not in self.bugs_resolution:
						self.bugs_resolution[resolution] = []
					self.bugs_resolution[resolution].append(bug)
		if wrt_feature == 'bugs':
			return self.bugs
		if wrt_feature == 'status':
			return self.bugs_status
		if wrt_feature == 'os':
			return self.bugs_os
		if wrt_feature == 'priority':
			return self.bugs_priority
		if wrt_feature == 'platform':
			return self.bugs_platform
		if wrt_feature == 'resolution':
			return self.bugs_resolution

	def __init__(self, connection):
		self.connection = connection
