from django import forms


class ImageUploadForm(forms.Form):
    image = forms.ImageField(
        widget=forms.FileInput(attrs={'class': "form-control"})
    )
