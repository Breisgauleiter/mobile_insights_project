/**
 * Home screen with video upload functionality.
 */
import React, {useState} from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  Alert,
} from 'react-native';
import {launchImageLibrary} from 'react-native-image-picker';
import type {NativeStackNavigationProp} from '@react-navigation/native-stack';
import type {RootStackParamList} from '../../App';
import {uploadVideo} from '../services/api';

type Props = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'Home'>;
};

export default function HomeScreen({navigation}: Props) {
  const [uploading, setUploading] = useState(false);

  const handlePickAndUpload = async () => {
    const result = await launchImageLibrary({
      mediaType: 'video',
      selectionLimit: 1,
    });

    if (result.didCancel || !result.assets?.length) {
      return;
    }

    const asset = result.assets[0];
    if (!asset.uri || !asset.fileName) {
      Alert.alert('Error', 'Could not read selected video.');
      return;
    }

    setUploading(true);
    try {
      const response = await uploadVideo(
        asset.uri,
        asset.fileName,
        asset.type ?? 'video/mp4',
      );
      Alert.alert('Upload Successful', `File: ${response.filename}`, [
        {
          text: 'View Results',
          onPress: () =>
            navigation.navigate('Results', {id: response.filename}),
        },
        {text: 'OK'},
      ]);
    } catch (err) {
      Alert.alert('Upload Failed', (err as Error).message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Mobile Insights</Text>
      <Text style={styles.subtitle}>Upload a gameplay video to detect highlights</Text>

      <TouchableOpacity
        style={[styles.button, uploading && styles.buttonDisabled]}
        onPress={handlePickAndUpload}
        disabled={uploading}>
        {uploading ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.buttonText}>Upload Video</Text>
        )}
      </TouchableOpacity>

      <TouchableOpacity
        style={styles.secondaryButton}
        onPress={() => navigation.navigate('Uploads')}>
        <Text style={styles.secondaryButtonText}>View All Uploads</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
    backgroundColor: '#0f0f23',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 14,
    color: '#888',
    marginBottom: 40,
    textAlign: 'center',
  },
  button: {
    backgroundColor: '#6c5ce7',
    paddingVertical: 16,
    paddingHorizontal: 48,
    borderRadius: 12,
    marginBottom: 16,
    minWidth: 200,
    alignItems: 'center',
  },
  buttonDisabled: {
    opacity: 0.6,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  secondaryButton: {
    paddingVertical: 12,
    paddingHorizontal: 32,
  },
  secondaryButtonText: {
    color: '#6c5ce7',
    fontSize: 14,
  },
});
