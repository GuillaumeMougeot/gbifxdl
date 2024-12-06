import paramiko

# Replace with your server details
hostname = "io.erda.au.dk"
port = 22
username = "gmo@ecos.au.dk"

# Set up the Paramiko client
try:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, port, username, password)  # Can also use key_filename for private key auth

    # Open an SFTP session
    sftp = ssh.open_sftp()
    
    # Perform SFTP operations
    sftp.put("local_file.txt", "remote_file.txt")  # Upload
    sftp.get("remote_file.txt", "local_file.txt")  # Download
    
    # Close the SFTP session
    sftp.close()
    ssh.close()
    print("SFTP operations completed successfully!")
except Exception as e:
    print(f"An error occurred: {e}")