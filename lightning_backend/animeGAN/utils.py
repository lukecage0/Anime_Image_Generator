from fastapi import HTTPException


class TimeoutException(HTTPException):
    def __init__(self, status_code=408, detail="Request timed out.", *args, **kwargs):
        super().__init__(status_code=status_code, detail=detail, *args, **kwargs)
