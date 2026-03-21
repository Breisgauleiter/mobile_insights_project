/**
 * API configuration for the Mobile Insights app.
 * Android emulator uses 10.0.2.2 to reach the host machine,
 * iOS simulator can use localhost directly.
 */
import {Platform} from 'react-native';

const DEFAULT_URL = Platform.select({
  android: 'http://10.0.2.2:3000',
  ios: 'http://localhost:3000',
  default: 'http://localhost:3000',
});

export const API_BASE_URL = DEFAULT_URL;
