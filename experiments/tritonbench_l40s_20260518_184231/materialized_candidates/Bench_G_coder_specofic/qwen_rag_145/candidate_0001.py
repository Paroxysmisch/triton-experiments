csharp
<Window x:Class="WPFContextMenu.MainWindow"
        xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        Title="Context Menu Example" Height="450" Width="800">
    <Grid>
        <Button x:Name="buttonWithContextMenu" Content="Right click here for a context menu" HorizontalAlignment="Left" VerticalAlignment="Top" Width="350" Height="50" Margin="28,85,0,0" FontSize="20" FontWeight="Bold"/>
        <!--Context Menu-->
        <ContextMenu x:Name="contextMenu">
            <MenuItem Header="Copy" Click="MenuItem_Click" />
            <MenuItem Header="Cut" Click="MenuItem_Click"/>
            <MenuItem Header="Paste" Click="MenuItem_Click"/>
        </ContextMenu>
    </Grid>
</Window>
