import asyncio, asyncssh, sys
from typing import TypedDict

class AsyncSFTPParams(TypedDict):
    host: str
    port: int
    username: str
    client_keys: list[str]

async def run_client(sftp_params: AsyncSFTPParams) -> None:
    async with asyncssh.connect(**sftp_params) as conn:
        async with conn.start_sftp_client() as sftp:
            await sftp.get('datasets/test3/1011881/1c41c4a0ed1dc2c62fda5f30f3844bddb0f66ed5.jpeg')
            await sftp.put('1c41c4a0ed1dc2c62fda5f30f3844bddb0f66ed5.jpeg', 'datasets/test5/1c41c4a0ed1dc2c62fda5f30f3844bddb0f66ed5.jpeg')

try:
    asyncio.run(run_client(sftp_params=AsyncSFTPParams(
            host="io.erda.au.dk",
            port=2222,
            username="gmo@ecos.au.dk",
            client_keys=["/mnt/c/Users/au761367/.ssh/id_rsa"])))
except (OSError, asyncssh.Error) as exc:
    sys.exit('SFTP operation failed: ' + str(exc))