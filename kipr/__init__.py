def sys_bgcolor(pyplot):
    """Reads system preference and sets plt background accordingly"""
    from winreg import ConnectRegistry, HKEY_CURRENT_USER, OpenKeyEx, QueryValueEx
    root = ConnectRegistry(None, HKEY_CURRENT_USER)
    policy_key = OpenKeyEx(root, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Themes\Personalize")
    light, _ = QueryValueEx(policy_key, "AppsUseLightTheme")
    if light:
        pyplot.style.use('default')
    else:
        pyplot.style.use('dark_background')