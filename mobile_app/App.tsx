/**
 * Mobile Insights App
 * Upload gameplay videos and view highlight detection results.
 */

import React from 'react';
import {NavigationContainer} from '@react-navigation/native';
import {createNativeStackNavigator} from '@react-navigation/native-stack';
import {StatusBar, useColorScheme} from 'react-native';

import HomeScreen from './src/screens/HomeScreen';
import UploadsScreen from './src/screens/UploadsScreen';
import ResultsScreen from './src/screens/ResultsScreen';

export type RootStackParamList = {
  Home: undefined;
  Uploads: undefined;
  Results: {id: string};
};

const Stack = createNativeStackNavigator<RootStackParamList>();

const screenOptions = {
  headerStyle: {backgroundColor: '#1e1e3a'},
  headerTintColor: '#fff',
  headerTitleStyle: {fontWeight: '600' as const},
};

function App() {
  const isDarkMode = useColorScheme() === 'dark';

  return (
    <NavigationContainer>
      <StatusBar barStyle={isDarkMode ? 'light-content' : 'dark-content'} />
      <Stack.Navigator screenOptions={screenOptions}>
        <Stack.Screen
          name="Home"
          component={HomeScreen}
          options={{title: 'Mobile Insights'}}
        />
        <Stack.Screen
          name="Uploads"
          component={UploadsScreen}
          options={{title: 'My Uploads'}}
        />
        <Stack.Screen
          name="Results"
          component={ResultsScreen}
          options={{title: 'Highlights'}}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

export default App;
