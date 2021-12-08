cd "C:\Users\$($env:UserName)\AppData\Local\Programs\Python\Python38"
if ($?) 
{
	python.exe -m pip install -i https://test.pypi.org/simple/  hopsy==0.2.0
	if (-Not ($?))
	{
		./python.exe -m pip install h5py
		./python.exe -m pip install cobra
		./python.exe -m pip install scipy
		./python.exe -m pip install numexpr
		./python.exe -m pip install -i https://test.pypi.org/simple/  hopsy==0.2.0
	}
	$HOPSY_VERSION=./python.exe -m pip freeze | Select-String -Pattern 'hopsy'
	if (-Not ($HOPSY_VERSION))
	{
		throw "Failed to install hopsy"
	}
}
else
{
	throw "C:\Users\$($env:UserName)\AppData\Local\Programs\Python\Python38 could not be accessed. Check if that directory exists. If not, download Python 3.8.0"
}
